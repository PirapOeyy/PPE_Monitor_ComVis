"""
multi_cam_yolo_metrics.py

Fitur:
- 4 kamera paralel (index USB atau URL RTSP).
- Overlay FPS per kamera + ringkas CPU/GPU.
- (Opsional) log CSV CPU/GPU/FPS per kamera tiap detik.

Install:
    pip install ultralytics opencv-python psutil
    # (opsional, untuk GPU%):
    pip install nvidia-ml-py3

Catatan:
- RTSP lebih stabil pakai backend FFMPEG; untuk USB cukup angka 0/1/2/3.
- HALF (FP16) hanya aktif kalau DEVICE cuda.
"""

from ultralytics import YOLO
import cv2, time, threading, csv
from pathlib import Path
import datetime as dt

# --- OPSIONAL monitoring ---
try:
    import psutil
except ImportError:
    psutil = None

# NVML untuk GPU utilization (NVIDIA)
try:
    from pynvml import (
        nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetUtilizationRates, nvmlDeviceGetMemoryInfo
    )
    _HAS_NVML = True
except Exception:
    _HAS_NVML = False

import torch

# ===================== KONFIG =====================
MODEL_PATH = r"D:\1st Rafi UPI\#SEM7\#MAGANG INDISMART\PPE_Monitor_ComVis\trainedModel.APD\weights\best.pt"

CONF = 0.40
IOU  = 0.50
IMG  = 640

# Pilih device otomatis: cuda kalau ada, else cpu
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
HALF   = True                           # aktifkan FP16 jika pakai GPU

# Sumber kamera: angka (USB) atau RTSP URL
SOURCES = [0, 1, 2, 3]
# Contoh RTSP:
# SOURCES = [
#   "rtsp://user:pass@192.168.1.10:554/stream1",
#   "rtsp://user:pass@192.168.1.11:554/stream1",
#   0, 1,
# ]

# Tuning
WINDOW_SCALE = 1.0                      # skala tampilan jendela
RECONNECT_WAIT = 2.0                    # detik saat reconnect
SHOW_WINDOW = True

# Logging CSV (opsional)
LOG_CSV = True
CSV_PATH = Path("metrics_log.csv")
CSV_INTERVAL = 1.0                      # detik
# ==================================================


def _is_cuda(dev: str) -> bool:
    return isinstance(dev, str) and dev.startswith("cuda")


class MetricsMonitor(threading.Thread):
    """
    Sampling CPU%, GPU% (NVML), mem GPU. Juga simpan FPS per kamera
    yang diupdate oleh thread kamera. Tulis ke CSV tiap interval.
    """
    def __init__(self, device_str: str, fps_dict: dict, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.dev = device_str
        self.fps_dict = fps_dict
        self.stop_event = stop_event
        self.gpu_index = 0
        self.nvml_handle = None
        self.last_csv = 0.0

        self.cpu = None
        self.gpu = None
        self.gpu_mem_used = None
        self.gpu_mem_total = None

        self.csv_ready = False
        if LOG_CSV:
            self._csv_init()

        # Init NVML jika perlu
        if _HAS_NVML and _is_cuda(self.dev):
            try:
                nvmlInit()
                self.nvml_handle = nvmlDeviceGetHandleByIndex(self.gpu_index)
            except Exception:
                self.nvml_handle = None

    def _csv_init(self):
        if not CSV_PATH.exists():
            with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                cols = ["timestamp", "cpu_percent", "gpu_percent", "gpu_mem_used_GB", "gpu_mem_total_GB", "camera", "fps"]
                w.writerow(cols)
        self.csv_ready = True

    def snapshot(self):
        # CPU
        if psutil:
            try:
                self.cpu = psutil.cpu_percent(interval=None)
            except Exception:
                self.cpu = None

        # GPU
        if _is_cuda(self.dev):
            # Utilization
            if self.nvml_handle is not None:
                try:
                    util = nvmlDeviceGetUtilizationRates(self.nvml_handle)
                    self.gpu = float(util.gpu)
                    mem = nvmlDeviceGetMemoryInfo(self.nvml_handle)
                    self.gpu_mem_used = mem.used / (1024**3)
                    self.gpu_mem_total = mem.total / (1024**3)
                except Exception:
                    self.gpu = None
            else:
                # fallback: pakai info mem Torch, tanpa persen util
                try:
                    idx = 0
                    self.gpu_mem_used = torch.cuda.memory_reserved(idx) / (1024**3)
                    self.gpu_mem_total = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
                except Exception:
                    self.gpu_mem_used = self.gpu_mem_total = None
                self.gpu = None
        else:
            self.gpu = None
            self.gpu_mem_used = self.gpu_mem_total = None

    def run(self):
        self.last_csv = time.time()
        while not self.stop_event.is_set():
            self.snapshot()

            # Tulis CSV tiap interval
            if LOG_CSV and self.csv_ready:
                now = time.time()
                if now - self.last_csv >= CSV_INTERVAL:
                    ts = dt.datetime.now().isoformat(timespec="seconds")
                    try:
                        with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
                            w = csv.writer(f)
                            for cam, fps in list(self.fps_dict.items()):
                                w.writerow([
                                    ts,
                                    None if self.cpu is None else f"{self.cpu:.1f}",
                                    None if self.gpu is None else f"{self.gpu:.1f}",
                                    None if self.gpu_mem_used is None else f"{self.gpu_mem_used:.3f}",
                                    None if self.gpu_mem_total is None else f"{self.gpu_mem_total:.3f}",
                                    cam,
                                    f"{fps:.2f}"
                                ])
                    except Exception:
                        pass
                    self.last_csv = now

            time.sleep(0.25)

        # Shutdown NVML
        if self.nvml_handle is not None:
            try:
                nvmlShutdown()
            except Exception:
                pass


class CamThread(threading.Thread):
    def __init__(self, src, name, model, metrics: MetricsMonitor, fps_dict: dict, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.src = src
        self.name = name
        self.model = model
        self.metrics = metrics
        self.fps_dict = fps_dict
        self.stop_event = stop_event
        self.cap = None
        self.fps = 0.0
        self._last_t = time.time()

    def _open(self):
        # RTSP lebih stabil dengan FFMPEG backend
        self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
        # Kurangi buffering RTSP (tidak semua backend taat)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return self.cap.isOpened()

    def _close(self):
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        self.cap = None

    def run(self):
        win = f"{self.name}"
        while not self.stop_event.is_set():
            if self.cap is None or not self.cap.isOpened():
                print(f"[{self.name}] membuka stream: {self.src}")
                if not self._open():
                    print(f"[{self.name}] gagal buka, retry {RECONNECT_WAIT}s")
                    time.sleep(RECONNECT_WAIT)
                    continue

            ok, frame = self.cap.read()
            if not ok or frame is None:
                print(f"[{self.name}] gagal read, reopen...")
                self._close()
                time.sleep(RECONNECT_WAIT)
                continue

            # === INFERENSI ===
            # Ultralytics API unified: kembalikan list Results
            results = self.model(
                frame,
                conf=CONF, iou=IOU, imgsz=IMG,
                device=DEVICE,
                half=(HALF and _is_cuda(DEVICE)),
                verbose=False
            )
            res = results[0]
            out = res.plot()

            # === FPS ===
            now = time.time()
            inst = 1.0 / max(now - self._last_t, 1e-6)
            self._last_t = now
            self.fps = inst if self.fps == 0 else (0.9 * self.fps + 0.1 * inst)
            self.fps_dict[self.name] = self.fps  # share ke monitor

            # === OVERLAY METRICS ===
            cpu_txt = "-" if self.metrics.cpu is None else f"{self.metrics.cpu:.0f}%"
            if self.metrics.gpu is None:
                if self.metrics.gpu_mem_used is not None and self.metrics.gpu_mem_total is not None:
                    gpu_txt = f"mem {self.metrics.gpu_mem_used:.2f}/{self.metrics.gpu_mem_total:.2f} GB"
                else:
                    gpu_txt = "-"
            else:
                # ada persen + mem
                if self.metrics.gpu_mem_used is not None and self.metrics.gpu_mem_total is not None:
                    gpu_txt = f"{self.metrics.gpu:.0f}% | {self.metrics.gpu_mem_used:.2f}/{self.metrics.gpu_mem_total:.2f} GB"
                else:
                    gpu_txt = f"{self.metrics.gpu:.0f}%"

            cv2.putText(out, f"{self.name} | FPS {self.fps:.1f}", (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
            cv2.putText(out, f"CPU {cpu_txt} | GPU {gpu_txt}", (10, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

            if WINDOW_SCALE != 1.0:
                out = cv2.resize(out, None, fx=WINDOW_SCALE, fy=WINDOW_SCALE)

            if SHOW_WINDOW:
                cv2.imshow(win, out)
                # tekan 'q' untuk keluar
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop_event.set()
                    break

        self._close()
        if SHOW_WINDOW:
            try:
                cv2.destroyWindow(win)
            except Exception:
                pass


def main():
    # Load model sekali
    model = YOLO(MODEL_PATH)
    # Optional: aktifkan cuDNN autotune
    if _is_cuda(DEVICE):
        torch.backends.cudnn.benchmark = True

    stop_event = threading.Event()
    fps_dict = {}

    # Monitor metrics global
    mon = MetricsMonitor(device_str=DEVICE, fps_dict=fps_dict, stop_event=stop_event)
    mon.start()

    # Worker per kamera
    workers = []
    for i, src in enumerate(SOURCES):
        name = f"Cam{i}"
        t = CamThread(src, name, model, mon, fps_dict, stop_event)
        t.start()
        workers.append(t)

    try:
        while any(w.is_alive() for w in workers):
            time.sleep(0.2)
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        stop_event.set()
        for w in workers:
            w.join()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()
