import cv2

def load_yolo_labels(label_path):
    yolo_bboxes = []
    with open(label_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            class_id = int(parts[0])
            center_x = float(parts[1])
            center_y = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            yolo_bboxes.append([class_id, center_x, center_y, width, height])
    return yolo_bboxes

def visualize_yolo_bounding_boxes(image_path, label_path):
    # 이미지를 읽어옵니다.
    image = cv2.imread(image_path)

    # YOLO 레이블을 로드합니다.
    yolo_bboxes = load_yolo_labels(label_path)

    # 바운딩 박스를 그립니다.
    height, width, _ = image.shape
    for bbox in yolo_bboxes:
        class_id, center_x, center_y, box_width, box_height = bbox
        
        # 좌표 변환
        x1 = int((center_x - box_width / 2) * width)
        y1 = int((center_y - box_height / 2) * height)
        x2 = int((center_x + box_width / 2) * width)
        y2 = int((center_y + box_height / 2) * height)
        
        # 바운딩 박스 그리기
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 초록색 박스
        cv2.putText(image, f'Class {class_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imwrite('del.jpg',image)

# 사용 예시
image_path = 'image_composite/Dataset.yolo/unannotated/images/data1.jpg'  # 이미지 경로
label_path = 'image_composite/Dataset.yolo/unannotated/labels/data1.txt'  # YOLO 레이블 파일 경로
visualize_yolo_bounding_boxes(image_path, label_path)
