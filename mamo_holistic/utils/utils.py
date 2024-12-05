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
        
def fig2data ( fig , close_fig = True):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )

    if close_fig:
        plt.close(fig)

    return buf

from PIL import Image

 
def fig2img ( fig, close_fig = True ):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data ( fig )
    w, h, d = buf.shape
    image =  Image.frombytes( "RGBA", ( w ,h ), buf.tostring( ) )

    if close_fig:
        plt.close(fig)

    return image


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', 'no', 'n', '0'}:
        return False
    elif value.lower() in {'true', 't', 'yes', 'y', '1'}:
        return True
    else:
        raise ValueError(f"{value} is not a valid boolean value")
    
    
    


def plot_roc_curve(fpr, tpr, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label='ROC curve')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    return fig, ax

def plot_pr_curve(precision, recall, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(recall, precision, label='PR curve')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    return fig, ax