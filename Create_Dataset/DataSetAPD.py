import yaml
import os
import cv2
from ultralytics import YOLO

#c:\ComVis_PPE_Monitor\DatasetTrain1_K3.yolov8
data_yaml = {
    'train' : 'c:\\ComVis_PPE_Monitor\\DatasetTrain1_K3.yolov8\\train',
    'val' : 'c:\\ComVis_PPE_Monitor\\DatasetTrain1_K3.yolov8\\valid',
    'nc' : 6,
    'names' :['Hardhat', 'No Hardhat', 'No Overall Suit', 'No Safety Boots', 'Overall Suit', 'Safety Boots']
}

with open('APD_Pertamina.yaml', 'w') as file:
    yaml.dump(data_yaml, file)
print("File APD_Pertamina.yaml berhasil dibuat.")
