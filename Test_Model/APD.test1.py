from ultralytics import YOLO
import cv2, time, threading
import torch
import torchvision

MODEL = YOLO('D:\\1st Rafi UPI\\#SEM7\\#MAGANG INDISMART\\PPE_Monitor_ComVis\\trainedModel.APD\\weights\\best.pt')
conf = 0.4
IOU = 0.5
IMG = 640
DEVICE = 'cpu'  # Ganti dengan 'cuda' jika menggunakan GPU
HALF = True

model = YOLO(MODEL)

cap = cv2.VideoCapture(0)  # Buka webcam

while True:
    ret, frame = cap.read()
    results = model(frame, stream=True)  # Deteksi objek
    for r in results:
        annotated_frame = r.plot()  # Gambar bounding box
        cv2.imshow('PPE Detetction', annotated_frame)
    if cv2.waitKey(1) == ord('q'):  # Tekan 'q' untuk keluar
        break

cap.release()
cv2.destroyAllWindows()
