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
   ```