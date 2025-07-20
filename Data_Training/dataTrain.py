from ultralytics import YOLO
import os
import yaml

model = YOLO('D:\\1st Rafi UPI\\#MAGANG INDISMART\\PPE_Monitoring\\Data_Training\\yolov8m.pt')
model.train(
    data='D:\\1st Rafi UPI\\#MAGANG INDISMART\\PPE_Monitoring\\Data_Training\\APD_K3.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    device='cpu'
)

model.export(format='onnx', dynamic=True)
# Save the model
model.save('D:\\1st Rafi UPI\\#MAGANG INDISMART\\PPE_Monitoring\\Data_Training\\yolo8m_apd_k3.onnx')