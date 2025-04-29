import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import torch
import matplotlib.pyplot as plt
import numpy as np
import wandb
import pathlib
import pandas as pd
from PIL import Image

# ----------------------------------------------------------------------------------------------

def show_mask_boundingbox(image_path, bb, ax = None):
    if isinstance(image_path, (pathlib.Path, str)):
        image_path = str(image_path)


        img = np.array(Image.open(image_path))
    else: #ndarray
        img = image_path

    bounding_boxes = pd.read_csv(bb)
    nombre_img = f"{img}.png"  # Concatenar la extensión
    bounding_boxes_grouped = bounding_boxes.groupby("id")
    group_df = bounding_boxes_grouped.get_group(nombre_img)

    boxes=[]

    # Iterar sobre las filas del DataFrame
    for index, row in group_df.iterrows():
        # Obtener los valores de las columnas
        x = row['x']
        y = row['y']
        w = row['w']
        h = row['h']
        
        xmin = x
        ymin = y
        xmax = x + w
        ymax = y + h
        boxes.append([xmin, ymin, xmax, ymax])

        lado1 = np.array([xmax-xmin, 0])  # Vector para el ancho (horizontal)
        lado2 = np.array([0, ymax-ymin])  # Vector para la altura (vertical)
        origen = np.array([xmin, ymin])  # Origen del rectángulo
        # Calcular las otras tres esquinas del rectángulo
        esquina1 = origen
        esquina2 = origen + lado1
        esquina3 = origen + lado1 + lado2
        esquina4 = origen + lado2
        x_values = [esquina1[0], esquina2[0], esquina3[0], esquina4[0], esquina1[0]]
        y_values = [esquina1[1], esquina2[1], esquina3[1], esquina4[1], esquina1[1]]


    # create overlay image, with mask semi-transparent in red
    if ax is None:
        plt.imshow(img, cmap='gray')
        plt.plot(x_values, y_values, 'r-')
        plt.axis('off')
        plt.show()
    else:
        ax.imshow(img, cmap='gray')
      #  ax.imshow(color_mask, alpha=0.4)
        ax.axis('off')


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


def show_batch_patch(batch, num_rows=2, bb=None):

    labels = batch[1].numpy()
    background_idx = np.where(labels == 0)[0]
    cancer_idx = np.where(labels == 1)[0]

    fig, axs = plt.subplots(num_rows, 2, figsize=(20, 5))

    for k in range(len(batch[0])):
        ax = axs[k // 2, k % 2] if num_rows > 1 else axs[k]
        image = batch[0][k].numpy()[0]
        show_mask_boundingbox(image, bb, ax=ax)


    # for row in range(num_rows):
    #     if row < len(background_idx):
    #         k = background_idx[row]
    #         image = batch[0][k].numpy()[0]
    #         mask = batch[2][k].numpy()
    #     else:
    #         image = np.zeros((224, 224))
    #         mask = np.zeros((224, 224))
    #         mask[60:120,60:120]=1
            
    #     show_mask_boundingbox(image, bb)
        
    #     if row < len(cancer_idx):
    #         k = cancer_idx[row]
    #         image = batch[0][k].numpy()[0]
    #         mask = batch[2][k].numpy()
    #     else:
    #         image = np.zeros((224, 224))
    #         mask = np.zeros((224, 224))
    #         mask[60:120,60:120]=1
        
    #     show_mask_boundingbox(image, bb)
        
    #     fig.suptitle("mean = {:.2f}, std = {:.2f}".format(batch[0].mean(), batch[0].std()))
    
    return fig2img(fig)

# ----------------------------------------------------------------------------------------------     
        
# def show_batch_ddsm(batch, num_rows=2):
#     labels = batch[1].numpy()
#     normal_idx = np.where(labels == 0)[0]
#     cancer_idx = np.where(labels == 1)[0]

#     fig, axs = plt.subplots(num_rows, 2, figsize=(10, 15))

#     for row in range(num_rows):
#         if row < len(normal_idx):
#             k = normal_idx[row]
#             image = batch[0][k].numpy()[0]
#             mask = batch[2][k].numpy()
#         else:
#             image = np.zeros((224, 224))
#             mask = np.zeros((224, 224))
#             mask[60:120,60:120]=1
            
#         show_mask_image(image, mask, ax=axs[row, 0], title='normal', multi_label=True)
        
#         if row < len(cancer_idx):
#             k = cancer_idx[row]
#             image = batch[0][k].numpy()[0]
#             mask = batch[2][k].numpy()
#         else:
#             image = np.zeros((224, 224))
#             mask = np.zeros((224, 224))
#             mask[60:120,60:120]=1
        
#         show_mask_image(image, mask, ax=axs[row, 1], title='cancer', multi_label=True)
        
#         fig.suptitle("mean = {:.2f}, std = {:.2f}".format(batch[0].mean(), batch[0].std()))
    
#     return fig2img(fig)
                 
# ----------------------------------------------------------------------------------------------

class VisualizeBatchPatchesCallback(Callback):
    def __init__(self, num_rows = 2, labels_file=None): # número de filas
        super().__init__()
        self.num_rows = num_rows
        self.labels_file = labels_file # path a los bounding boxes

    def on_train_epoch_start(self, trainer, pl_module):
        # Get the first batch from the training dataloader
        train_dataloader = trainer.train_dataloader
        
        batch = next(iter(train_dataloader))
        #images, labels, masks = batch
        
        # Show the batch
        img = show_batch_patch(batch, num_rows=self.num_rows, bb=self.labels_file) # Tengo que cambiar el yaml 
        experiment = trainer.logger.experiment
        experiment.log({"training batch": wandb.Image(img)})
        
        
        return super().on_train_epoch_start(trainer, pl_module)
        
        
    def on_validation_epoch_start(self, trainer, pl_module):
        # Get the first batch from the validation dataloader
        val_dataloader = trainer.val_dataloaders
        batch = next(iter(val_dataloader))
        #images, labels, masks = batch
        
        img = show_batch_patch(batch, num_rows=self.num_rows, bb=self.labels_file) # Tengo que cambiar el yaml
        experiment = trainer.logger.experiment
        experiment.log({"validation batch": wandb.Image(img)})
        
        
        
        return super().on_validation_epoch_start(trainer, pl_module)
    
# class VisualizeBatchImagesCallback(Callback):
#     def __init__(self, num_rows = 2):
#         super().__init__()
#         self.num_rows = num_rows

#     def on_train_epoch_start(self, trainer, pl_module):
#         # Get the first batch from the training dataloader
#         train_dataloader = trainer.train_dataloader
        
#         batch = next(iter(train_dataloader))
#         #images, labels, masks = batch
        
#         # Show the batch
#         img = show_batch_ddsm(batch, num_rows=self.num_rows)
#         experiment = trainer.logger.experiment
#         experiment.log({"training batch": wandb.Image(img)})
        
        
#         return super().on_train_epoch_start(trainer, pl_module)
        
        
#     def on_validation_epoch_start(self, trainer, pl_module):
#         # Get the first batch from the validation dataloader
#         val_dataloader = trainer.val_dataloaders
#         batch = next(iter(val_dataloader))
#         #images, labels, masks = batch
        
#         img = show_batch_ddsm(batch, num_rows=self.num_rows)
#         experiment = trainer.logger.experiment
#         experiment.log({"validation batch": wandb.Image(img)})
        
        
        
#         return super().on_validation_epoch_start(trainer, pl_module)



            
