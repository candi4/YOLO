import os
import shutil
import random
import yaml
from tqdm import tqdm

from utils import root_dir


def split_data(unannotated_data_dir, train_dir, val_dir, test_dir,
               train_ratio, val_ratio, test_ratio):
    """

    Parameters
    ----------
    - unannotated_data_dir: unannotated_data_dir/images, unannotated_data_dir/labels
    - train_ratio + val_ratio + test_ratio = 1
    """
    os.makedirs(os.path.join(train_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(test_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(test_dir, 'labels'), exist_ok=True)

    images = os.listdir(os.path.join(unannotated_data_dir, 'images'))
    labels = [os.path.splitext(f)[0]+'.txt' for f in images]
    
    # Shuffle
    data = list(zip(images, labels))
    random.shuffle(data)

    # Split
    dataset_size = len(data)
    train_endindex = int(dataset_size * train_ratio)
    val_endindex = train_endindex + int(dataset_size * val_ratio)
    train_data = data[:train_endindex]
    val_data = data[train_endindex:val_endindex]
    test_data = data[val_endindex:]

    # Copy files
    for image, label in tqdm(train_data, desc='train_data'):
        shutil.copy(os.path.join(unannotated_data_dir, 'images', image), os.path.join(train_dir, 'images', image))
        shutil.copy(os.path.join(unannotated_data_dir, 'labels', label), os.path.join(train_dir, 'labels', label))
    for image, label in tqdm(val_data, desc='val_data'):
        shutil.copy(os.path.join(unannotated_data_dir, 'images', image), os.path.join(val_dir, 'images', image))
        shutil.copy(os.path.join(unannotated_data_dir, 'labels', label), os.path.join(val_dir, 'labels', label))
    for image, label in tqdm(test_data, desc='test_data'):
        shutil.copy(os.path.join(unannotated_data_dir, 'images', image), os.path.join(test_dir, 'images', image))
        shutil.copy(os.path.join(unannotated_data_dir, 'labels', label), os.path.join(test_dir, 'labels', label))

    print(f"Data split into {len(train_data)} train, {len(val_data)} val, and {len(test_data)} test samples.")
    print(f"Total data is {dataset_size}")


split_data(unannotated_data_dir='_Dataset.yolo/unannotated', 
           train_dir='_Dataset.yolo/train', val_dir='_Dataset.yolo/val', test_dir='_Dataset.yolo/test',
           train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)


data = {
    'path': os.path.join(root_dir, 'image_composite', '_Dataset.yolo'),
    'train': 'train/images',
    'val': 'val/images',
    'test': 'test/images',
    'names': ["Module"]
}
with open('_Dataset.yolo/data.yaml', 'w') as file:
    yaml.dump(data, file, default_flow_style=False)

test_data = {
    'path': os.path.join(root_dir, 'image_composite', '_Dataset.yolo'),
    'val': 'test/images',
    'names': ["Module"]
}
with open('_Dataset.yolo/testdata.yaml', 'w') as file:
    yaml.dump(test_data, file, default_flow_style=False)