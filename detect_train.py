from ultralytics import YOLO
from Utils.utils import root_dir


# Load a model
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
# n s m l x

# Train the model with GPUs
results = model.train(data=f"{root_dir}/Dataset/Finger detection via sementation.v5i.yolov8/data.yaml", epochs=10000, imgsz=640, device=None,
                      save_period=100)

