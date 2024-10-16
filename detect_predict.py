from ultralytics import YOLO

# Load a model
model = YOLO("/home/hojun/project/YOLO/runs/detect/train/weights/best.pt")  # pretrained model

# Run batched inference on a list of images
results = model('/home/hojun/project/YOLO/ezgif-6-13eb75ace9-jpg', stream=False)  # return a generator of Results objects

# Process results generator
i = 0
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    # result.show()  # display to screen
    result.save(filename=f"result/result{i}.jpg")  # save to disk
    
    
    print(i)
    print(boxes.conf)
    print(boxes.xyxy)
    boxes.conf.argmax(dim=None, keepdim=False) 
    
    
    print()
    i += 1