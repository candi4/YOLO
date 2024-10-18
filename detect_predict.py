from ultralytics import YOLO
import os
from Utils.utils import root_dir

# Load a model
model = YOLO("model/detect.pt")  # pretrained model

# Run batched inference on a list of images
results = model(f'{root_dir}/RawImage/ezgif-6-13eb75ace9-jpg', stream=True)  # return a generator of Results objects

# Process results generator
i = 0
for result in results:
    print(result.orig_img.shape)
    
    os.makedirs("_result/detect", exist_ok=True)
    result.save(filename=f"_result/detect/result{i}.jpg")  # save to disk
    
    boxes = result.boxes  # Boxes object for bounding box outputs
    
    
    print(i)
    print(boxes.conf)
    print(boxes.xyxy)
    print(boxes.conf.argmax(dim=None, keepdim=False))
    
    
    print()
    i += 1