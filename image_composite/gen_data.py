import yaml
import time
import os
import numpy as np

from utils import GenYoloData

with open("parameters.yaml", "r") as file:
    params = yaml.safe_load(file)

# Print the not used images
bg_dir = params['bg_dir']
exist_bg_files = set([f"{bg_dir}/{path}" for path in os.listdir(bg_dir) if os.path.isfile(f"{bg_dir}/{path}")])
using_bg_files = set([f"{bg_dir}/{path}" for path in params['background']])
unused_bg_filenames = using_bg_files - exist_bg_files
print("Unsed backgrounds:")
if len(unused_bg_filenames) > 0:
    for filename in list(unused_bg_filenames):
        print("   ",filename)
else:
    print("    Nothing")

# Use every object images
obj_dir = params['obj_dir']
obj_filenames = [path for path in os.listdir(obj_dir) if os.path.isfile(f"{obj_dir}/{path}")]

data_order = 0
while True:
    bg_filename = np.random.choice(list(params['background'].keys()))
    bg_config = params['background'][bg_filename]
    print(bg_filename)
    obj_filename = np.random.choice(obj_filenames)
    print(obj_filename)
    genyolo = GenYoloData(bg_filename=f'{bg_dir}/{bg_filename}',
                        obj_filename=f'{obj_dir}/{obj_filename}')
    for i in range(3):
        start_individual = time.time()
        dataname = f'data{data_order}'
        genyolo.combine(bg_scale=bg_config['bg_scale'], obj_scale=bg_config['obj_scale'], obj_range=bg_config['obj_range'],
                        bg_weight=bg_config['bg_weight'], obj_weight=bg_config['obj_weight'], gamma=bg_config['gamma'],
                        )
        crop_shape = np.random.randint(*bg_config['crop_shape_range'], size=2)
        genyolo.crop(crop_shape=crop_shape)
        genyolo.save(directory='Dataset.yolo/unannotated/', dataname=dataname)
        print(f"{data_order}th data saved: {dataname}")
        print("    Individual consumed time:", time.time()-start_individual)
        data_order += 1

print("YAML files saved successfully!")
print("Total consumed time:", time.time()-start_main)