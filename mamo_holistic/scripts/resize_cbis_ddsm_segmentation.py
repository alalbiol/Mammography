import pathlib
from PIL import Image

import pandas as pd

import argparse
from tqdm import tqdm

import numpy as np








if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Resize DDSM images.')
    parser.add_argument('--dataset_root_folder', type=str, default='/home/alalbiol/Data/mamo/CBIS-DDSM-segmentation' , help='Path to the root folder of DDSM images.')
    parser.add_argument('--dest_root', type=str, default = '/home/alalbiol/Data/mamo/' , help='Path to the destination root folder for resized images.')
    parser.add_argument('--width', type=int, default= 1792, help='Width to resize images to.')
    parser.add_argument('--height', type=int, default= 2240, help='Height to resize images to.')
    
    args = parser.parse_args()
    
    ddsm_root_folder = pathlib.Path(args.dataset_root_folder)
    dest_root = pathlib.Path(args.dest_root)
    width = args.width
    height = args.height
    
    

    dest_root = pathlib.Path(dest_root) / f'CBIS-DDSM-segmentation-{height}x{width}'
    
    print("dest_root: ", dest_root)
    
    if not dest_root.exists():
        dest_root.mkdir(parents=True)
        (dest_root / 'images').mkdir()
        (dest_root / 'masks').mkdir()
    
    
    bouding_boxes = pd.read_csv(ddsm_root_folder / 'bounding_boxes.csv')
    
    print("Number of bounding boxes: ", len(bouding_boxes))
    
    scaled_bouding_boxes = []
    
    for k, row in bouding_boxes.iterrows():
        id = row['id']
        x = row['x']
        y = row['y']
        w = row['w']
        h = row['h']
        label = row['label']
        
        image = Image.open(ddsm_root_folder / f'images/{id}.png')
        mask = Image.open(ddsm_root_folder / f'masks/{id}.png')
        
        
        #print(np.array(image).max(), np.array(mask).max())
        
        scalex = width / image.width
        scaley = height / image.height
        
        x = int(x * scalex)
        y = int(y * scaley)
        w = int(w * scalex)
        h = int(h * scaley)
        
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        mask = mask.resize((width, height), Image.Resampling.NEAREST)
        
        mask_np = np.array(mask)
        assert mask_np.sum() > 0, f"Mask is empty for image {id}"
        

        image = np.array(image)
        image = np.clip(image, 0, 65535)

        # Convert back to an image if needed
        image = Image.fromarray(image.astype(np.uint16))
 
 
        scaled_bouding_boxes.append({'id': id, 'x': x, 'y': y, 'w': w, 'h': h, 'label': label})
        
        if k % 1 == 0:
            print(f"Saving image {k} with id {id}")
            
        image.save(dest_root / f'images/{id}.png')
        mask.save(dest_root / f'masks/{id}.png')
        
        #print(np.array(image).max(), np.array(mask).max())
        
        
    scaled_bouding_boxes = pd.DataFrame(scaled_bouding_boxes)
    scaled_bouding_boxes.to_csv(dest_root / 'bounding_boxes.csv', index=False)
            
        