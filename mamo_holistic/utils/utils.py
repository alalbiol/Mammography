import pathlib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def show_mask_image(image_path, mask_path,ax = None, points = None):
    if isinstance(image_path, (pathlib.Path, str)):
        image_path = str(image_path)
        img = np.array(Image.open(image_path))
    else: #ndarray
        img = image_path
        
    if mask_path is None:
        mask = np.zeros_like(img, dtype=np.uint8)
    elif isinstance(mask_path, (pathlib.Path, str)):
        mask = np.array(Image.open(mask_path))
    else: #ndarray
        mask = mask_path
    
    # Create an RGB version of the mask with only the red channel active
    red_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    red_mask[:, :, 0] = mask  # Set the red channel to the mask values

    
    # create overlay image, with mask semi-transparent in red
    if ax is None:
        plt.imshow(img, cmap='gray')
        plt.imshow(red_mask, alpha=0.4)
        plt.axis('off')
    else:
        ax.imshow(img, cmap='gray')
        ax.imshow(red_mask, alpha=0.4)
        ax.axis('off')
        
    if points is not None:
        x = points[:,0]
        y = points[:,1]
        ax.plot(x, y, 'b')
        