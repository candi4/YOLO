from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
# n s m l x

# Train the model with 2 GPUs
results = model.train(data="Finger detection via sementation.v5i.yolov8/data.yaml", epochs=100, imgsz=640, device=[0, 1])