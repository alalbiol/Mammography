import sys
import os.path
import pathlib
import numpy as np
import matplotlib.pyplot as plt


ddsm_dir = pathlib.Path('/home/Data/mamo/DDSM_png')
dcm_images = list(ddsm_dir.glob('**/*.png'))
dcm_overlay = list(ddsm_dir.glob('**/*.OVERLAY'))
print(dcm_images[0])

def has_overlay(image_path):
    overlay_path = image_path.with_suffix('.OVERLAY')
    return overlay_path.exists()

for image_path in dcm_images:
    if has_overlay(image_path):
        print(f"{image_path} has an associated overlay.")
    else:
        print(f"{image_path} does not have an associated overlay.")

