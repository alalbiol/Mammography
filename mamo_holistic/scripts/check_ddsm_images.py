import pathlib
import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def check_images(ddsm_images):
    
    all_data = []
    
    for image_path in tqdm(ddsm_images):
        image_data = {'image_path': image_path}
        image = np.array(Image.open(image_path))
        image_data['image_std'] = np.std(image)
        image_data['image_mean'] = np.mean(image)
        image_data['image_max'] = np.max(image)
        image_data['image_min'] = np.min(image)
        image_data['image_contrast'] = image_data['image_max'] - image_data['image_min']
        
        all_data.append(image_data)

    df = pd.DataFrame(all_data)
    print(df.head())
    df.to_csv("ddsm_images_stats.csv", index=False)


def main():
    ddsm_root = pathlib.Path("/media/HD/mamo/DDSM_png_noclipping")
    ddsm_images = list(ddsm_root.glob("**/*.png"))
    print("Number of DDSM images: ", len(ddsm_images))  
    
    check_images(ddsm_images)
    
    
if __name__ == "__main__":
    main()