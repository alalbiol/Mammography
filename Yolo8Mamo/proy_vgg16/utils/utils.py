import pathlib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt



def show_mask_image(image_path, mask_path,ax = None, points = None, title = None,multi_label = False):
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
    
    if multi_label == False:
        # Create an RGB version of the mask with only the red channel active
        color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        color_mask[:, :, 0] = mask  # Set the red channel to the mask values
    else:
        # Create an RGB version of the mask with red and green channels active
        color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        color_mask[mask == 1, 1] = 255  # Set the green channel to 255 where mask is 1 (benign mass)
        color_mask[mask == 2, 1] = 128  # Set the green channel to 128 where mask is 2 (bening calcification)
        color_mask[mask == 3, 0] = 255  # Set the grereden channel to 255 where mask is 3 (malignant mass)
        color_mask[mask == 4, 0] = 128  # Set the red channel to 128 where mask is 4 (malignant calcification)
    
    # create overlay image, with mask semi-transparent in red
    if ax is None:
        plt.imshow(img, cmap='gray')
        plt.imshow(color_mask, alpha=0.4)
        plt.axis('off')
        if title is not None:
            plt.title(title)
    else:
        ax.imshow(img, cmap='gray')
        ax.imshow(color_mask, alpha=0.4)
        ax.axis('off')
        if title is not None:
            ax.set_title(title)
        
    if points is not None:
        x = points[:,0]
        y = points[:,1]
        ax.plot(x, y, 'b')