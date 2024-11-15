import os
import shutil
import random






def split_yolo_data(source_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    # 디렉토리 생성
    os.makedirs(os.path.join(source_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(source_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(source_dir, 'test'), exist_ok=True)

    # 이미지 및 라벨 파일 목록 가져오기
    images = [f for f in os.listdir(source_dir) if f.endswith('.jpg') or f.endswith('.png')]
    labels = [f.replace('.jpg', '.txt').replace('.png', '.txt') for f in images]

    # 이미지와 라벨 쌍을 섞기
    data = list(zip(images, labels))
    random.shuffle(data)

    # 데이터 분할
    total = len(data)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    # 파일 복사
    for dataset, name in zip([train_data, val_data, test_data], ['train', 'val', 'test']):
        for img, lbl in dataset:
            shutil.copy(os.path.join(source_dir, img), os.path.join(source_dir, name, img))
            shutil.copy(os.path.join(source_dir, lbl), os.path.join(source_dir, name, lbl))

    print(f"Data split into {len(train_data)} train, {len(val_data)} val, and {len(test_data)} test samples.")

# 사용 예시
source_directory = 'path_to_your_yolo_dataset'  # YOLO 데이터셋 경로
split_yolo_data(source_directory)
