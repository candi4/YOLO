from ultralytics import YOLO


# Load a model
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
# n s m l x

# Train the model with GPUs
results = model.train(data=f"image_composite/_Dataset.yolo/data.yaml", epochs=1000, imgsz=640, device=[0,1,2,3],
                      save_period=10, patience=100, batch=600)

