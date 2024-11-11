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


sys.path.append('../lib')
from annotation_utils import read_annotation_image

def get_overlay_file(png_image):
    """Returns true if there exist a corresponding OVERALY file for a png_image

    Args:
        png_image (Path): image to check

    Returns:
        boolean: true if overlay exists
    """
    overlay_file = pathlib.Path(str(png_image).replace('.png', '.OVERLAY'))
    return overlay_file if overlay_file.exists() else None


def resize_image(image, target_size=(1152,896), annotations=None):
    """Resize an image to target_size, if ddsm annotations exits are also
       resscaled, in particular bounding_box and outline. 
       Interpolation used is cv2.INTER_LANCZOS4

    Args:
        image (np.array): image to resize
        target_size (tuple): height x width
        annotations : ddsm annotations for image. New annotations are created, with scaled bb and outline. 
                The rest of annotations remain unchanged. Defaults to None.
    """
   
    # target_size = (1152,896) # (height x width)
    original_size = (image.shape[0], image.shape[1]) # (height x width)
    scale_x = target_size[1] / original_size[1]
    scale_y = target_size[0] / original_size[0]
    
    if annotations is not None:
        new_annotations = []
        for annotation in annotations:
            new_annotation = annotation.copy()
            
            #[xmin, ymin, xmax, ymax, centerx, centery, width, height]
            boundary = new_annotation['outline']
            new_boundary = (boundary[0]* scale_x, boundary[1]*scale_y)
            new_bounding_box = new_annotation['bounding_box'].copy()
            
        
            #[xmin, ymin, xmax, ymax, centerx, centery, width, height]
            for i in [0, 2, 4, 6]:  # Indices for x values
                new_bounding_box[i] *= scale_x
            for i in [1, 3, 5, 7]:  # Indices for y values
                new_bounding_box[i] *= scale_y


            new_annotation['outline'] = new_boundary
            new_annotation['bounding_box'] = new_bounding_box
            
            
            new_annotations.append(new_annotation)
    
 
    resized_image = cv2.resize(image, (target_size[1], target_size[0])   , interpolation = cv2.INTER_LANCZOS4)

    if annotations is not None:
        return resized_image, new_annotations
    
    return resized_image


def create_abnormality_mask(image, outline):
    """Create a mask for the abnormality defined by the outline. 
    The mask is a binary image with the same shape as the input image, 
    where the pixels inside the outline are set to 255 and the rest to 0.
    

    Args:
        image (np.array): image used to get size of mask
        outline (tuple): (x,y) coordinates of the outline of the abnormality

    Returns:
        np.array: mask of the abnormality
    """
    # 
    mask = np.zeros(image.shape, dtype=np.uint8)
    
    # Combine x and y coordinates into a single array of shape (n, 2)
    coords = np.array(list(zip(outline[0], outline[1]))).reshape(-1, 1, 2).astype(np.int32)
    
    # Fill the mask with the polygon defined by the outline
    mask = cv2.fillPoly(mask, [coords], 255)
    
    return mask


def create_ddsm_df(ddsm_root_folder):
    """Creates dataframe with all the images in the ddsm_root_folder and their corresponding overlay files.
    If there is no overlay file for an image, the overlay_path is set to None.   

    Args:
        ddsm_root_folder (Path): root folder of the DDSM images

    Returns:
        Dataframe: two columns: original_path and overlay_path
    """
    
    
    if isinstance(ddsm_root_folder,str):
        ddsm_root_folder = pathlib.Path(ddsm_root_folder)
    
    ddsm_images = list(ddsm_root_folder.glob('**/*.png'))
    print("Number of images: ", len(ddsm_images))

    ddsm_df = pd.DataFrame(ddsm_images, columns=['original_path'])
    ddsm_df['overlay_path'] = ddsm_df['original_path'].apply(get_overlay_file)


    return ddsm_df



def rescale_all_ddsm(ddsm_df, target_size=(1152,896), dest_root_folder = pathlib.Path('/home/alalbiol/Data/mamo/')):
    
    all_annotations = []
    for i in tqdm(range(len(ddsm_df))):
        
        annotated_image = ddsm_df.iloc[i]['original_path']
        
        image = np.array(Image.open(annotated_image))
        
        if ddsm_df.iloc[i]['overlay_path'] is not None:
            annotations = read_annotation_image(ddsm_df.iloc[i]['overlay_path'])
            image_resized, annotations_resized = resize_image(image, target_size, annotations)
        else:
            annotations = None    
            image_resized = resize_image(image, target_size)
            annotations_resized = None
        
        dest_path = dest_root_folder / ddsm_df.iloc[i]['original_path'].relative_to(ddsm_root_folder)
        
        if dest_path.parent.exists() is False:
            dest_path.parent.mkdir(parents=True)
        
        image_resized = Image.fromarray(image_resized)
        image_resized.save(str(dest_path))
        
        #print("dest_path: ", dest_path)
        
            
        image_id = "/".join(dest_path.parts[-4:])
        
        if annotations_resized is not None:
            for k, annotation_resized in enumerate(annotations_resized):
                annotation_resized['image_id'] = image_id
                type = annotation_resized['type'] # MASS or CALCIFICATION
                pathology = annotation_resized['pathology'] # MALIGNANT or BENIGN of BENIGN_WITHOUT_CALLBACK
                if "BENIGN" in pathology:
                    pathology = "BENIGN" # BENIGN_WITHOUT_CALLBACK is considered BENIGN
                
                annotation_resized['mask_id'] = f'{image_id.replace(".png", "")}_{type}_{pathology}_mask_{k}.png'
                
                mask = create_abnormality_mask(np.array(image_resized), annotation_resized['outline'])
                
                mask_path = dest_root_folder / f'{annotation_resized["mask_id"]}'
                mask_image = Image.fromarray(mask)
                mask_image.save(str(mask_path))
                #print(mask_path)
                
                all_annotations.append(annotation_resized)
        
    

    all_annotations = pd.DataFrame(all_annotations)            
    return all_annotations

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Resize DDSM images.')
    parser.add_argument('--ddsm_root_folder', type=str, default='/media/HD/mamo/DDSM_png' , help='Path to the root folder of DDSM images.')
    parser.add_argument('--dest_root', type=str, default = '/home/alalbiol/Data/mamo/' , help='Path to the destination root folder for resized images.')
    parser.add_argument('--width', type=int, default= 896, help='Width to resize images to.')
    parser.add_argument('--height', type=int, default= 1152, help='Height to resize images to.')
    parser.add_argument('--jsonout', type=str, default= 'ddsm_annotations.json', help='File to save all the annotations.')
    
    args = parser.parse_args()
    
    ddsm_root_folder = args.ddsm_root_folder
    dest_root = args.dest_root
    width = args.width
    height = args.height
    
    
    ddsm_df = create_ddsm_df(ddsm_root_folder)
    dest_root = pathlib.Path(dest_root) / f'DDSM_png_2_{height}x{width}'
    
    all_annotations = rescale_all_ddsm(ddsm_df, target_size=(height,width), 
                                       dest_root_folder = pathlib.Path(dest_root), 
                                       )
    
    
    # Save all annotations to a json file
    all_annotations.to_json(args.jsonout, orient='records', lines=True)
    
    #all_annotations.to_csv(args.csvout, index=False)
    
    
    
    
    # You can now use ddsm_root_folder, dest_root, width, and height in your code