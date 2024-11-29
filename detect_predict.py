from ultralytics import YOLO
import os
from Utils.utils import root_dir
from tqdm import tqdm
import shutil

model_dir = 'model'
model_filenames = os.listdir(model_dir)

if os.path.exists('_result/detect'):
    shutil.rmtree('_result/detect')

for model_filenames in tqdm(model_filenames):
    # Load a model
    model = YOLO(os.path.join(model_dir,model_filenames))  # pretrained model
    print('model_filenames',model_filenames)

    # Run batched inference on a list of images
    results = model('testimage', stream=True)  # return a generator of Results objects

    # Process results generator
    i = 0
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        if len(boxes) > 0:
            os.makedirs("_result/detect", exist_ok=True)
            result.save(filename=f"_result/detect/{model_filenames}_{os.path.basename(result.path)}")  # save to disk
            print(boxes.conf)
        
        i += 1