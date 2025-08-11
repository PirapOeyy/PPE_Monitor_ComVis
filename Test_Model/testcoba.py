from ultralytics import YOLO
import cv2

model = YOLO('runs\\detect\\train2\\weights\\best.pt')

image_path = "C:\\ComVis_PPE_Monitor\\DatasetTrain1_K3.yolov8\\test\\images\\sharepoint_elnusa_3660_jpg.rf.7903068d6e2b6f3c3db35c9fe52e4fc8.jpg" 
img = cv2.imread(image_path)  # Buka webcam

results = model.predict(source=img, conf=0.4)  

# === Ambil hasil prediksi (bounding boxes, class, dll) ===
for result in results:
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Koordinat box
        cls_id = int(box.cls[0])               # Class index
        conf = float(box.conf[0])              # Confidence
        label = model.names[cls_id]            # Nama class

        
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imshow("YOLOv8 Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


