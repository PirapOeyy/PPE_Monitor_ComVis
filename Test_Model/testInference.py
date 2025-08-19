import cv2
from ultralytics import YOLO

# Load model YOLO hasil training sendiri
model = YOLO("runs\\detect\\train5\\weights\\best.pt")

# Baca gambar
img = cv2.imread("D:\\1st Rafi UPI\\#SEM7\\#MAGANG INDISMART\\PPE_Monitor_ComVis\\sharepoint_elnusa_3411_jpg.rf.30a3897b8d0d2c7a0d31d1a6522fd9e5.jpg")# Jalankan prediksi
resized_img = cv2.resize(img, (640, 640))

results = model(resized_img, conf=0.5, iou=0.7)

# Visualisasi hasil deteksi
test_img = results[0].plot() 

# Tampilkan gambar hasil deteksi
cv2.imshow('PPE Detection', test_img)
cv2.waitKey(0)
cv2.destroyAllWindows() 


#Person:
# "D:\\1st Rafi UPI\\#MAGANG INDISMART\\zz.Folder.tidak dipakai\\Raw_Dataset\\12.FullPerson\\pers3.jpg"
#  D:\\1st Rafi UPI\\#MAGANG INDISMART\\zz.Folder.tidak dipakai\\Raw_Dataset\\12.FullPerson\\pers.jpg
#SepatuSafety: 

#SendalNoSafety: 
#"D:\1st Rafi UPI\\#MAGANG INDISMART\\zz.Folder.tidak dipakai\\Raw_Dataset\\7.SafetyShoes\\IMG_20250805_125625497_BURST0018.jpg"
# "D:\\1st Rafi UPI\\#MAGANG INDISMART\\zz.Folder.tidak dipakai\\Raw_Dataset\\7.SafetyShoes\\IMG_20250805_125712_244.jpg"
#Pertamina: 
#"D:\\1st Rafi UPI\\#MAGANG INDISMART\\PPE_Monitor_ComVis\\sharepoint_elnusa_3411_jpg.rf.30a3897b8d0d2c7a0d31d1a6522fd9e5.jpg
# D:\\1st Rafi UPI\\#MAGANG INDISMART\\zz.Folder.tidak dipakai\\Raw_Dataset\\2.SafetyGoggle_dataset\\safetyglass25.jpeg