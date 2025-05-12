import lightning as L
from lightning.pytorch.callbacks import Callback
import torch
import matplotlib.pyplot as plt
import numpy as np
import wandb
import pathlib
import pandas as pd
from PIL import Image



# ----------------------------------------------------------------------------------------------

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

# ----------------------------------------------------------------------------------------------
def show_box_image(image_path, box_path,ax = None, title = None):
    img=image_path
    box=box_path['boxes'].numpy()[0]
    
    if ax is None:
        plt.imshow(img)
        plot_bbox(box)
        plt.axis('off')
        if title is not None:
            plt.title(title)
    else:
        ax.imshow(img)
        plot_bbox(box)
        ax.axis('off')
        if title is not None:
            ax.set_title(title)
# ----------------------------------------------------------------------------------------------
def show_batch_ddsm(batch, num_rows=1):
    images, boxes = batch

    fig, axs = plt.subplots(num_rows, 1, figsize=(8, 4 * num_rows))

    if num_rows == 1:
        axs = [axs]  # para que axs sea iterable

    for row in range(num_rows):
        if row < len(images):
            image = images[row][0]  # suponemos que es 1 canal (CxHxW)
            box = boxes[row]
        else:
            # Relleno en caso de que pidamos más filas de las que hay en el batch
            image = np.zeros((640, 640))
            box = {'boxes': torch.zeros((0, 4)), 'labels': torch.zeros((0,))}

        show_box_image(image, box, ax=axs[row])
    
    return fig2img(fig)

# ----------------------------------------------------------------------------------------------
def plot_bbox(box):
    
    xmin, ymin, xmax, ymax = box   
        
    # Dibujar el rectángulo
    plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], 'r')

# ----------------------------------------------------------------------------------------------


class VisualizeBatchImagesCallback(Callback):
    def __init__(self, num_rows = 2):
        super().__init__()
        self.num_rows = num_rows

    def on_train_epoch_start(self, trainer, pl_module):
        # Get the first batch from the training dataloader
        train_dataloader = trainer.train_dataloader
        
        batch = next(iter(train_dataloader))
        #images, labels, boxs = batch
        
        # Show the batch
        img = show_batch_ddsm(batch, num_rows=self.num_rows)
        experiment = trainer.logger.experiment
        experiment.log({"training batch": wandb.Image(img)})
        
        
        return super().on_train_epoch_start(trainer, pl_module)
        
        
    def on_validation_epoch_start(self, trainer, pl_module):
        # Get the first batch from the validation dataloader
        val_dataloader = trainer.val_dataloaders
        batch = next(iter(val_dataloader))
        #images, labels, boxs = batch
        
        img = show_batch_ddsm(batch, num_rows=self.num_rows)
        experiment = trainer.logger.experiment
        experiment.log({"validation batch": wandb.Image(img)})
        
        
        
        return super().on_validation_epoch_start(trainer, pl_module)
    



            
