#from ultralytics import YOLO


#model = YOLO("yolo11n.pt")


#train_results = model.train(data="C:/Users/admin/Desktop/AI2/demo_custom_yolo11/LicensePlateDataset/data.yaml", epochs=1, imgsz=140)

from ultralytics import YOLO

model = YOLO("runs/detect/train4/weights/best.pt")

results = model("test_images", save = True)
results[0].show()