# YOLO
How to use YOLO

## Train YOLO model
1. Make image dataset using [roboflow](https://roboflow.com/) 
2. Download dataset as `YOLOv8` format
3. Install `ultralytics`, `cuda`, and `pytorch`
   ```shell
   conda create -n yolo python=3.8 -y
   conda activate yolo
   conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia -y
   pip install ultralytics
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
      result.show()  # display to screen
      result.save(filename=f"result/result{i}.jpg")  # save to disk
   ```
   * The `str` in `model` can be folder name or filename. The file can a video or an image. Refer to [ultralytics/predict#inference-sources](https://docs.ultralytics.com/modes/predict/#inference-sources).
   * If `stream` is false, it calculates at one time. Else, it calculates one time for one loop in `for result in results`.
   * You can refer [ultralytics/predict](https://docs.ultralytics.com/modes/predict) documentation.
   * The description of attributes like `boxes.xyxy` can be found in [ultralytics/result](https://docs.ultralytics.com/reference/engine/results/).
## Generate training data
1. Move images into `image_composite/raw_images/background` and `image_composite/raw_images/object`.
2. Write parameters into `image_composite/parameters.yaml`.
3. Run `image_composite/parameter_test.py` while revising parameters. 
   Whenever run the code, it automatically deletes `image_composite/test/`.
   * `bg_scale`: Scale background image.
   * `obj_scale`: Scale object image.
   * `obj_range`: Area for object to be included in.
   * `bg_weight`: Weight for background image in weighted sum.
   * `obj_weight`: Weight for object image in weighted sum.
   * `gamma`: 0-255 value to be added in weighted sum.
   * `crop_shape_range`: The length of each side of the cropped image is randomly determined within this range. The smallest and largest images is saved in `image_composite\test\images`.
4. Run `split_data.py` to split dataset into train, validate, test dataset.
* References:
   * [Object Detection Datasets Overview](https://docs.ultralytics.com/datasets/detect/)

## Inspect data
`detect_inspect`