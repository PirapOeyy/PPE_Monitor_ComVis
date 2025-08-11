from ultralytics import YOLO
import os
import yaml
from multiprocessing import freeze_support

def main():
    model = YOLO("C:\\ComVis_PPE_Monitor\\PPE_Monitor_ComVis\\Data_Training\\yolov8m.pt")
    model.train(
    data="C:\\ComVis_PPE_Monitor\\PPE_Monitor_ComVis\\APD_Pertamina.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0
)

    model.export(format='onnx', dynamic=True)

if __name__ == '__main__':
    freeze_support()
    main()

# Save the model
#model.save('yolo8m_apd_k3.onnx')