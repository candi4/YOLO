import yaml
import time
import os
import numpy as np
import shutil

from utils import GenYoloData

output_dir = '_samples/'
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

with open("parameters.yaml", "r") as file:
    params = yaml.safe_load(file)

for bg_filename in params['background'].keys():
    print(f"{bg_filename} start...")
    bg_dir = params['bg_dir']
    bg_config = params['background'][bg_filename]

    obj_dir = params['obj_dir']
    obj_filenames = [path for path in os.listdir(obj_dir) if os.path.isfile(f"{obj_dir}/{path}")]
    obj_filename = np.random.choice(obj_filenames)
    obj_filename = '(ry0)(rx179)(delx0.1).png'

    genyolo = GenYoloData(bg_filename=f'{bg_dir}/{bg_filename}',
                        obj_filename=f'{obj_dir}/{obj_filename}')

    for i in range(2):
        filename, file_extension = os.path.splitext(bg_filename)
        dataname = f"{filename}_{i}.{file_extension}"

        genyolo.combine(bg_scale=bg_config['bg_scale'], obj_scale=bg_config['obj_scale'], obj_range=bg_config['obj_range'],
                        bg_weight=bg_config['bg_weight'], obj_weight=bg_config['obj_weight'], gamma=bg_config['gamma'],
                        regionimg_filename=f'{output_dir}/region/{bg_filename}'
                        )

        crop_shape = bg_config['crop_shape_range']
        genyolo.crop(crop_shape=np.ones(2, dtype=int)*crop_shape[i])
        
        genyolo.save(directory=output_dir, dataname=dataname)

print("Done!!!")