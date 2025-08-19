# multi_cam_yolo4.py
# Jalankan: python multi_cam_yolo4.py
# ESC atau 'q' untuk keluar.

from ultralytics import YOLO
import cv2, time, threading, os
import numpy as np
import torch

# ==== GPU metrics (NVML) ====
try:
    from pynvml import (
        nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetUtilizationRates, nvmlDeviceGetMemoryInfo,
        nvmlDeviceGetTemperature, NVML_TEMPERATURE_GPU
    )
    HAS_NVML = True
except Exception:
    HAS_NVML = False

# ================== KONFIGURASI ==================
MODEL_PATH = "D:\\1st Rafi UPI\\#SEM7\\#MAGANG INDISMART\\PPE_Monitor_ComVis\\trainedModel.APD\\weights\\best.pt"
CONF = 0.40
IOU  = 0.50
IMG  = 640

# Otomatis pilih GPU jika ada
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
HALF   = True                        # hanya aktif di CUDA
INFER_EVERY = 1                      # jalankan deteksi tiap N frame (1 = setiap frame)

# Sumber kamera: index USB (0,1,2,3) atau URL RTSP
SOURCES = [0, 1, 2, 3]
# Contoh RTSP:
# SOURCES = [
#   "rtsp://user:pass@192.168.1.10:554/stream1",
#   "rtsp://user:pass@192.168.1.11:554/stream1",
#   "rtsp://user:pass@192.168.1.12:554/stream1",
#   "rtsp://user:pass@192.168.1.13:554/stream1",
# ]

# Tampilan
SHOW_MODE = "grid"                   # "grid" = 1 jendela mozaik 2x2, "many" = 4 jendela
GRID_TILE = (640, 360)               # ukuran tiap ubin saat mode grid
WINDOW_SCALE = 1.0                   # skala display (1.0 = asli)
RECONNECT_WAIT = 2.0                 # detik tunggu saat reconnect
# =================================================


def is_cuda(dev: str) -> bool:
    return isinstance(dev, str) and dev.startswith("cuda")


# ----------- Monitor GPU global (util/mem/temp) -----------
class GPUMonitor(threading.Thread):
    def __init__(self, device_str: str, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.dev = device_str
        self.stop_event = stop_event
        self.gpu_index = 0
        self.handle = None
        self.util = None       # persen
        self.mem_used = None   # GB
        self.mem_total = None  # GB
        self.temp = None       # C

    def run(self):
        use_nvml = HAS_NVML and is_cuda(self.dev)
        if use_nvml:
            try:
                nvmlInit()
                self.handle = nvmlDeviceGetHandleByIndex(self.gpu_index)
            except Exception:
                use_nvml = False
                self.handle = None

        while not self.stop_event.is_set():
            if use_nvml and self.handle is not None:
                try:
                    u = nvmlDeviceGetUtilizationRates(self.handle)
                    self.util = float(u.gpu)
                    m = nvmlDeviceGetMemoryInfo(self.handle)
                    self.mem_used  = m.used / (1024**3)
                    self.mem_total = m.total / (1024**3)
                    self.temp = float(nvmlDeviceGetTemperature(self.handle, NVML_TEMPERATURE_GPU))
                except Exception:
                    self.util = self.mem_used = self.mem_total = self.temp = None
            else:
                # CPU mode atau NVML tidak tersedia
                self.util = self.temp = None
                try:
                    if is_cuda(self.dev):
                        idx = 0
                        self.mem_used  = torch.cuda.memory_reserved(idx) / (1024**3)
                        self.mem_total = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
                    else:
                        self.mem_used = self.mem_total = None
                except Exception:
                    self.mem_used = self.mem_total = None

            time.sleep(0.25)

        if self.handle is not None and HAS_NVML:
            try: nvmlShutdown()
            except Exception: pass


# ---------------- Worker per kamera ----------------
class CamWorker(threading.Thread):
    def __init__(self, src, name, model, gpu_mon: GPUMonitor, stop_event: threading.Event, bus: dict, bus_lock: threading.Lock):
        super().__init__(daemon=True)
        self.src = src
        self.name = name
        self.model = model
        self.gpu_mon = gpu_mon
        self.stop_event = stop_event
        self.bus = bus
        self.bus_lock = bus_lock
        self.cap = None
        self.fps = 0.0
        self._last_t = time.time()
        self._frame_id = 0

    def _open(self):
        # USB index → backend default; RTSP → FFMPEG lebih stabil
        if isinstance(self.src, int):
            self.cap = cv2.VideoCapture(self.src)
        else:
            self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
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

            self._frame_id += 1
            do_infer = (self._frame_id % max(1, INFER_EVERY) == 0)

            if do_infer:
                results = self.model(
                    frame,
                    conf=CONF, iou=IOU, imgsz=IMG,
                    device=DEVICE,
                    half=(HALF and is_cuda(DEVICE)),
                    verbose=False
                )
                res = results[0]
                out = res.plot()
            else:
                out = frame

            # FPS smoothing
            now = time.time()
            inst = 1.0 / max(now - self._last_t, 1e-6)
            self._last_t = now
            self.fps = inst if self.fps == 0 else 0.9 * self.fps + 0.1 * inst

            # Overlay GPU metrics
            util = "-" if self.gpu_mon.util      is None else f"{self.gpu_mon.util:.0f}%"
            mem  = "-" if self.gpu_mon.mem_used  is None else f"{self.gpu_mon.mem_used:.2f}/{self.gpu_mon.mem_total:.2f} GB"
            temp = "-" if self.gpu_mon.temp      is None else f"{self.gpu_mon.temp:.0f}°C"

            cv2.putText(out, f"{self.name} | FPS {self.fps:.1f}",
                        (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
            cv2.putText(out, f"GPU {util} | {mem} | {temp}",
                        (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

            # kirim ke display bus
            if WINDOW_SCALE != 1.0:
                out = cv2.resize(out, None, fx=WINDOW_SCALE, fy=WINDOW_SCALE)

            with self.bus_lock:
                self.bus[self.name] = out

        self._close()


# --------------- Thread khusus display ---------------
class DisplayThread(threading.Thread):
    def __init__(self, mode, bus, bus_lock, stop_event):
        super().__init__(daemon=True)
        self.mode = mode
        self.bus = bus
        self.bus_lock = bus_lock
        self.stop_event = stop_event

    def _show_many(self, frames):
        # Satu window per kamera
        for name, frm in frames.items():
            if frm is None: continue
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            cv2.imshow(name, frm)

    def _grid(self, frames):
        names = sorted(frames.keys())
        tiles = []
        for name in names[:4]:
            img = frames.get(name, None)
            if img is None:
                img = np.zeros((GRID_TILE[1], GRID_TILE[0], 3), dtype=np.uint8)
            else:
                img = cv2.resize(img, GRID_TILE)
            tiles.append(img)
        while len(tiles) < 4:
            tiles.append(np.zeros((GRID_TILE[1], GRID_TILE[0], 3), dtype=np.uint8))
        top = np.hstack((tiles[0], tiles[1]))
        bot = np.hstack((tiles[2], tiles[3]))
        grid = np.vstack((top, bot))
        cv2.namedWindow("MultiCam", cv2.WINDOW_NORMAL)
        cv2.imshow("MultiCam", grid)

    def run(self):
        if self.mode == "off":
            return
        while not self.stop_event.is_set():
            with self.bus_lock:
                frames = dict(self.bus)
            if self.mode == "many":
                if frames: self._show_many(frames)
            else:  # grid
                if frames: self._grid(frames)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                self.stop_event.set()
                break
            time.sleep(0.005)
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


def main():
    # Load YOLO sekali
    model = YOLO(MODEL_PATH)
    if is_cuda(DEVICE):
        torch.backends.cudnn.benchmark = True

    stop_event = threading.Event()

    # GPU monitor global
    gpu_mon = GPUMonitor(device_str=DEVICE, stop_event=stop_event)
    gpu_mon.start()

    # Shared bus untuk display
    frame_bus = {}
    bus_lock = threading.Lock()

    # Start worker kamera
    workers = []
    for i, src in enumerate(SOURCES):
        w = CamWorker(src, f"Cam{i}", model, gpu_mon, stop_event, frame_bus, bus_lock)
        w.start()
        workers.append(w)

    # Start display
    disp = DisplayThread(SHOW_MODE, frame_bus, bus_lock, stop_event)
    disp.start()

    try:
        while any(w.is_alive() for w in workers):
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        for w in workers: w.join()
        disp.join()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()
