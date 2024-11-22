import yaml
import time
import os
import numpy as np
from tqdm import tqdm
import shutil

from utils import GenYoloData

start_main = time.time()

with open("parameters.yaml", "r") as file:
    params = yaml.safe_load(file)

# Print the not used images
bg_dir = params['bg_dir']
exist_bg_files = set([f"{bg_dir}/{path}" for path in os.listdir(bg_dir) if os.path.isfile(f"{bg_dir}/{path}")])
using_bg_files = set([f"{bg_dir}/{path}" for path in params['background']])
unused_bg_filenames = exist_bg_files - using_bg_files
print("Unused backgrounds:")
if len(unused_bg_filenames) > 0:
    for filename in list(unused_bg_filenames):
        print("   ",filename)
else:
    print("    Nothing")

# Use every object images
obj_dir = params['obj_dir']
obj_filenames = [path for path in os.listdir(obj_dir) if os.path.isfile(f"{obj_dir}/{path}")]


if os.path.exists('_Dataset.yolo/unannotated/'):
    shutil.rmtree('_Dataset.yolo/unannotated/')

data_num = 1_000
with tqdm(total=data_num) as pbar:
    obj_dir = params['obj_dir']
    obj_filenames = [path for path in os.listdir(obj_dir) if os.path.isfile(f"{obj_dir}/{path}")]
    data_order = 0
    while data_order < data_num:
        bg_filename = np.random.choice(list(params['background'].keys()))
        bg_config = params['background'][bg_filename]
        obj_filename = np.random.choice(obj_filenames)
        genyolo = GenYoloData(bg_filename=f'{bg_dir}/{bg_filename}',
                            obj_filename=f'{obj_dir}/{obj_filename}')
        for i in range(1):
            dataname = f'data{data_order}'
            genyolo.combine(bg_scale=bg_config['bg_scale'], obj_scale=bg_config['obj_scale'], obj_range=bg_config['obj_range'],
                            bg_weight=bg_config['bg_weight'], obj_weight=bg_config['obj_weight'], gamma=bg_config['gamma'],
                            )
            crop_shape = np.random.randint(*bg_config['crop_shape_range'], size=2)
            genyolo.crop(crop_shape=crop_shape)
            genyolo.save(directory='_Dataset.yolo/unannotated/', dataname=dataname)
            data_order += 1; pbar.update(1)

print("Generated dataset is saved successfully!")
print("Total consumed time:", time.time()-start_main)