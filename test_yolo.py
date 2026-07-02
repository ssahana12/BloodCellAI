from ultralytics import YOLO

print("Loading model...")

model = YOLO("yolov8n.pt")

print("Running detection...")

results = model.predict(
    source="datasets/bccd/train/img/BloodImage_00001.jpeg",
    save=True,
    conf=0.25
)

print("Finished!")
