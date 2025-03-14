import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import torch

import matplotlib.pyplot as plt
import numpy as np
from utils.utils import show_mask_image
from utils.utils import fig2img
import wandb


def show_batch(batch, num_rows=2):
    labels = batch[1].numpy()
    background_idx = np.where(labels == 0)[0]
    benign_mass_idx = np.where(labels == 1)[0]
    benign_calc_idx = np.where(labels == 2)[0]
    malig_mass_idx = np.where(labels == 3)[0]
    malig_calc_idx = np.where(labels == 4)[0]

    fig, axs = plt.subplots(num_rows, 4, figsize=(20, 5))

    for row in range(num_rows):
        # if row < len(background_idx):
        #     k = background_idx[row]
        #     image = batch[0][k].numpy()[0]
        #     mask = batch[1][k].numpy()
        # else:
        #     image = np.zeros((224, 224))
        #     mask = np.zeros((224, 224))
        #     mask[60:120,60:120]=1
            
        # show_mask_image(image, mask, ax=axs[row, 0], title='background')
        
        if row < len(benign_mass_idx):
            k = benign_mass_idx[row]
            image = batch[0][k].numpy()[0]
            mask = batch[1][k].numpy()
        else:
            image = np.zeros((224, 224))
            mask = np.zeros((224, 224))
            mask[60:120,60:120]=1
        
        show_mask_image(image, mask, ax=axs[row, 0], title='benign_mass')
        
        if row < len(benign_calc_idx):
            k = benign_calc_idx[row]
            image = batch[0][k].numpy()[0]
            mask = batch[1][k].numpy()
        else:
            image = np.zeros((224, 224))
            mask = np.zeros((224, 224))
            mask[60:120,60:120]=1
            
        show_mask_image(image, mask, ax=axs[row, 1], title='benign_calc')
        
        if row < len(malig_mass_idx):
            k = malig_mass_idx[row]
            image = batch[0][k].numpy()[0]
            mask = batch[1][k].numpy()
        else:
            image = np.zeros((224, 224))
            mask = np.zeros((224, 224))
            mask[60:120,60:120]=1
            
        show_mask_image(image, mask, ax=axs[row, 2], title='malig_mass')
        
        if row < len(malig_calc_idx):
            k = malig_calc_idx[row]
            image = batch[0][k].numpy()[0]
            mask = batch[1][k].numpy()
        else:
            image = np.zeros((224, 224))
            mask = np.zeros((224, 224))
            mask[60:120,60:120]=1
            
        show_mask_image(image, mask, ax=axs[row, 3], title='malig_calc')
        
        fig.suptitle("mean = {:.2f}, std = {:.2f}".format(batch[0].mean(), batch[0].std()))
    
    return fig2img(fig)


class VisualizeBatchImagesCallback(Callback):
    def __init__(self, num_rows = 2):
        super().__init__()
        self.num_rows = num_rows

    def on_train_epoch_start(self, trainer, pl_module):
        # Get the first batch from the training dataloader
        train_dataloader = trainer.train_dataloader
        
        batch = next(iter(train_dataloader))
        #images, labels, masks = batch
        
        # Show the batch
        img = show_batch(batch, num_rows=self.num_rows)
        experiment = trainer.logger.experiment
        experiment.log({"training batch": wandb.Image(img)})
        
        
        return super().on_train_epoch_start(trainer, pl_module)
        
        
    def on_validation_epoch_start(self, trainer, pl_module):
        # Get the first batch from the validation dataloader
        val_dataloader = trainer.val_dataloaders
        batch = next(iter(val_dataloader))
        #images, labels, masks = batch
        
        img = show_batch(batch, num_rows=self.num_rows)
        experiment = trainer.logger.experiment
        experiment.log({"validation batch": wandb.Image(img)})
        
        
        
        return super().on_validation_epoch_start(trainer, pl_module)
