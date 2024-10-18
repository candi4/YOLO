from ultralytics import YOLO
import os
from Utils.utils import root_dir

# Load a model
model = YOLO("model/segment.pt")  # pretrained model

# Run batched inference on a list of images
results = model(f'{root_dir}/RawImage/ezgif-6-13eb75ace9-jpg', stream=True)  # return a generator of Results objects

print(dir(results))

# Process results generator
i = 0
for result in results:
    print(result.orig_img.shape)
    
    os.makedirs("_result/segment", exist_ok=True)
    result.save(filename=f"_result/segment/result{i}.jpg")  # save to disk
    
    boxes = result.boxes
    masks = result.masks
    
    
    print(i)
    print(masks.data.shape)
    print(len(masks.xy))
    print(boxes.conf.argmax(dim=None, keepdim=False))
    print(masks.orig_shape)
    
    print()
    i += 1