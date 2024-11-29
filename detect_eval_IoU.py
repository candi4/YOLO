from ultralytics import YOLO
from tqdm import tqdm
from pathlib import Path
import numpy as np
import cv2  # 이미지 크기 확인을 위해 추가

# Load YOLO model
model = YOLO("/home/hojun/project/YOLO/runs/detect/train2/weights/best.pt")  # 학습된 모델 경로

# Set the path for the test datase
test_images_dir = Path('/home/hojun/project/YOLO/image_composite/_Dataset.yolo/test/images')  # 문자열을 Path 객체로 변환
test_labels_dir = Path('/home/hojun/project/YOLO/image_composite/_Dataset.yolo/test/labels')  # 문자열을 Path 객체로 변환



# IoU calculation function
def calculate_iou(box1, box2):
    # box: [x_min, y_min, x_max, y_max]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Area of both boxes - intersection
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    return intersection / union if union > 0 else 0

# Evaluate the test dataset
ious = []

for image_path in tqdm(sorted(test_images_dir.glob('*.jpg'))):
    img = cv2.imread(str(image_path))
    img_height, img_width = img.shape[:2]

    results = model.predict(image_path, conf=0.25)
    pred_boxes = results[0].boxes.xyxy.cpu().numpy() if len(results) > 0 else []

    # Load GT labels
    label_path = test_labels_dir / (image_path.stem + '.txt')
    if not label_path.exists():
        print(f"Warning: Label file {label_path} not found!")
        continue

    with open(label_path, 'r') as f:
        gt_boxes = []
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"Warning: Incorrect label format in {label_path}")
                continue
            _, x_center, y_center, width, height = map(float, parts)

            # Convert normalized values to pixel values
            x_min = (x_center - width / 2) * img_width
            y_min = (y_center - height / 2) * img_height
            x_max = (x_center + width / 2) * img_width
            y_max = (y_center + height / 2) * img_height
            gt_boxes.append([x_min, y_min, x_max, y_max])

    # Calculate IoU
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        print(f"Warning: No predictions or ground truth for {image_path}")
        continue

    for gt_box in gt_boxes:
        max_iou = 0
        for pred_box in pred_boxes:
            iou = calculate_iou(gt_box, pred_box)
            max_iou = max(max_iou, iou)
        ious.append(max_iou)

# Calculate mean IoU
mean_iou = np.mean(ious)
print(f"Mean IoU: {mean_iou}")
