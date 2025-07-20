import cv2
import os
import yaml
from ultralytics import YOLO

model = YOLO('yolo8m.pt')


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Tidak dapat akses kamera.")
    exit()

# Config File Path
config_path = 'D:\\1st Rafi UPI\\#MAGANG INDISMART\\Pertamina_APDchecker\\data.yaml'
if not os.path.exists(config_path):
    print(f"File konfigurasi tidak ditemukan: {config_path}")
    exit()

