import yaml
import time
import os
import numpy as np

from utils import GenYoloData

with open("parameters.yaml", "r") as file:
    params = yaml.safe_load(file)

# Lists the not used images
bg_dir = 'raw_images/background/'
exist_bg_files = set([f"{bg_dir}/{path}" for path in os.listdir(bg_dir) if os.path.isfile(f"{bg_dir}/{path}")])
using_bg_files = set([f"{bg_dir}/{path}" for path in params['bg_filenames']])
unused_bg_filenames = using_bg_files - exist_bg_files
print("Unsed backgrounds:")
if len(unused_bg_filenames) > 0:
    for filename in list(unused_bg_filenames):
        print("   ",filename)
else:
    print("    Nothing")

obj_dir = 'raw_images/object/'
obj_filenames = [path for path in os.listdir(obj_dir) if os.path.isfile(f"{obj_dir}/{path}")]
print(obj_filenames)

data_order = 0
while True:

    bg_filename = np.random.choice(params['bg_filenames'])
    print(bg_filename)
    obj_filename = np.random.choice(obj_filenames)
    print(obj_filename)
    genyolo = GenYoloData(bg_filename=f'{bg_dir}/{bg_filename}',
                        obj_filename=f'{obj_dir}/{obj_filename}')
    for i in range(3):
        start_individual = time.time()
        dataname = f'data{data_order}'
        genyolo.combine(bg_scale=8.0, obj_scale=0.8, obj_range=[[120,120],  # x1 y1
                                                                [470,450]], # x2 y2
                        bg_weight=0.6, obj_weight=1.3, gamma=0.,
                        )
        crop_shape = np.random.randint(1500,2000,size=(2))
        genyolo.crop(crop_shape=crop_shape)
        genyolo.save(directory='Dataset.yolo/unannotated/', dataname=dataname)
        print(f"{data_order}th data saved: {dataname}")
        print("    Individual consumed time:", time.time()-start_individual)
        data_order += 1

print("YAML files saved successfully!")
print("Total consumed time:", time.time()-start_main)