# YOLO
How to use YOLO

## How to train YOLO model
1. Make image dataset using [roboflow](https://roboflow.com/) 
2. Download dataset as `YOLOv8` format
3. Install `ultralytics`, `cuda`, and `pytorch`
   ```shell
   conda create -n yolo python=3.8 -y
   conda activate yolo
   conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia -y
   pip install -r requirements.txt
   ```
4. Train
   ```python
   from ultralytics import YOLO
   model = YOLO("yolo11n.pt")
   results = model.train(data="/the/directory/of/data.yaml", epochs=100, imgsz=640)
   ```
   You can refer [ultralytics/train](https://docs.ultralytics.com/modes/train) documentation.
5. Predict
   ```python
   from ultralytics import YOLO

   # Load a model
   model = YOLO("best.pt")  # pretrained model

   # Run batched inference on a list of images
   results = model('foldername/or/filename/of/video/or/image', stream=False)  # return a generator of Results objects

   # Process results generator
   i = 0
   for result in results:
      boxes = result.boxes  # Boxes object for bounding box outputs
      result.show()  # display to screen
      result.save(filename=f"result/result{i}.jpg")  # save to disk
   ```
   * The `str` in `model` can be folder name or filename. The file can a video or an image.
   * If `stream` is false, it calculates at one time. Else, it calculates one time for one loop in `for result in results`.
   * You can refer [ultralytics/predict](https://docs.ultralytics.com/modes/predict) documentation.
