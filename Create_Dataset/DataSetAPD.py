import yaml
import os
import cv2
from ultralytics import YOLO
from skimage.io import imread

data_yaml = {
    'train' : 'D:\\1st Rafi UPI\\#MAGANG INDISMART\\PPE.dataset.yolov8\\train',
    'val' : 'D:\\1st Rafi UPI\\#MAGANG INDISMART\\PPE.dataset.yolov8\\valid',
    'nc' : 11,
    'names' : ['Head', 'MaskRespi', 'NoHarness', 'Person', 'SafetyBoots', 'SafetyEarmuffs',
               'SafetyGlasses', 'SafetyGloves', 'SafetyHarness', 'SafetyHelm', 'SafetyVest']
}

with open('APD_K3.yaml', 'w') as file:
    yaml.dump(data_yaml, file)
print("File APD_K3.yaml berhasil dibuat.")
