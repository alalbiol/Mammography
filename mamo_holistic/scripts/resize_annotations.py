import pathlib
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2  
import os
import sys  
import pandas as pd

import argparse
from tqdm import tqdm

# this file is the same as resize_ddsm.py, but ONLY process annotations
# to generate a json with all annotations


sys.path.append('../utils')
from utils.ddsm_resize_utils import create_ddsm_df, rescale_all_ddsm






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Resize DDSM images.')
    parser.add_argument('--ddsm_root_folder', type=str, default='/media/HD/mamo/DDSM_png' , help='Path to the root folder of DDSM images.')
    parser.add_argument('--dest_root', type=str, default = '/home/alalbiol/Data/mamo/' , help='Path to the destination root folder for resized images.')
    parser.add_argument('--width', type=int, default= 896, help='Width to resize images to.')
    parser.add_argument('--height', type=int, default= 1152, help='Height to resize images to.')
    parser.add_argument('--csvout', type=str, default= 'ddsm_annotations.csv', help='File to save all the annotations.')
    
    args = parser.parse_args()
    
    ddsm_root_folder = args.ddsm_root_folder
    dest_root = args.dest_root
    width = args.width
    height = args.height
    
    
    ddsm_df = create_ddsm_df(ddsm_root_folder)
    dest_root = pathlib.Path(dest_root) / f'DDSM_png_2_{height}x{width}'
    
    all_annotations = rescale_all_ddsm(ddsm_df, target_size=(height,width), 
                                    dest_root_folder = pathlib.Path(dest_root), 
                                    use_image_magick=False)
    
    
    # Save all annotations to a json file
    all_annotations.to_json(args.csvout, orient='records', lines=True)
    
    #all_annotations.to_csv(args.csvout, index=False)
    
    
    
    
    # You can now use ddsm_root_folder, dest_root, width, and height in your code