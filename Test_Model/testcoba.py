from ultralytics import YOLO
import cv2

model = YOLO('yolov11m.pt')
cap = cv2.VideoCapture(0)  # Buka webcam

while True:
    ret, frame = cap.read()
    results = model(frame, stream=True)  # Deteksi objek
    for r in results:
        annotated_frame = r.plot()  # Gambar bounding box
        cv2.imshow('YOLO', annotated_frame)
    if cv2.waitKey(1) == ord('q'):  # Tekan 'q' untuk keluar
        break

cap.release()
cv2.destroyAllWindows()