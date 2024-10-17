import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import os.path
import matplotlib.image as mpimg

#get path of this notebook

current_path = pathlib.Path().absolute()
print(current_path)

#Para que lea de otras carpetas
sys.path.append(str(current_path / '../prepare_data'))
sys.path.append(str(current_path / '../datasets'))
sys.path.append(str(current_path / '../pruebas_lucia'))

ddsm_dir = pathlib.Path('/home/Data/mamo/DDSM_png')
dcm_images = list(ddsm_dir.glob('**/*.png'))
dcm_overlay = list(ddsm_dir.glob('**/*.OVERLAY'))

from funciones_lucia import create_bbox
from funciones_lucia import read_overlay, get_outline_curve, has_overlay


for image_path in dcm_images:
    overlay_path = image_path.with_suffix('.OVERLAY')
    image_name = image_path.stem
    three_folders_above = image_path.parents[2].name+"."+image_path.parents[1].name+"."+image_path.parents[0].name
    labels_dir = '/home/lloprib/proyecto_mam/Mammography/Yolo8Mamo/func_pruebas_lucia/labels/'
    #labels_dir.mkdir(exist_ok=True)
    txt_path = labels_dir + three_folders_above +'.'+f"{image_name}.txt"
    if has_overlay(image_path):
        overlay = read_overlay(overlay_path)
        bbox = create_bbox(overlay)
        with open(txt_path, 'w') as txt_file:
            txt_file.write(f'{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}')
    else:
        with open(txt_path, 'w') as txt_file:
            txt_file.write('')

print(len(dcm_images))
print(len(dcm_overlay))
