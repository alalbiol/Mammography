import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import torch

class LearningRateWarmUpCallback(Callback):
    def __init__(self, warmup_epochs, initial_lr, max_lr):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.initial_lr = initial_lr
        self.max_lr = max_lr

    def on_train_epoch_start(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        if current_epoch < self.warmup_epochs:
            # Calculate the new learning rate
            new_lr = self.initial_lr + (self.max_lr - self.initial_lr) * (current_epoch / self.warmup_epochs)
            for param_group in trainer.optimizers[0].param_groups:
                param_group['lr'] = new_lr
            print(f"Epoch {current_epoch + 1}: Learning rate set to {new_lr}")



import matplotlib.pyplot as plt
import numpy as np
from utils.utils import show_mask_image
from utils.utils import fig2img
import wandb


def show_batch_patch(batch, num_rows=2):
    labels = batch[1].numpy()
    background_idx = np.where(labels == 0)[0]
    benign_mass_idx = np.where(labels == 1)[0]
    benign_calc_idx = np.where(labels == 2)[0]
    malig_mass_idx = np.where(labels == 3)[0]
    malig_calc_idx = np.where(labels == 4)[0]

    fig, axs = plt.subplots(num_rows, 5, figsize=(20, 5))

    for row in range(num_rows):
        if row < len(background_idx):
            k = background_idx[row]
            image = batch[0][k].numpy()[0]
            mask = batch[2][k].numpy()
        else:
            image = np.zeros((224, 224))
            mask = np.zeros((224, 224))
            mask[60:120,60:120]=1
            
        show_mask_image(image, mask, ax=axs[row, 0], title='background')
        
        if row < len(benign_mass_idx):
            k = benign_mass_idx[row]
            image = batch[0][k].numpy()[0]
            mask = batch[2][k].numpy()
        else:
            image = np.zeros((224, 224))
            mask = np.zeros((224, 224))
            mask[60:120,60:120]=1
        
        show_mask_image(image, mask, ax=axs[row, 1], title='benign_mass')
        
        if row < len(benign_calc_idx):
            k = benign_calc_idx[row]
            image = batch[0][k].numpy()[0]
            mask = batch[2][k].numpy()
        else:
            image = np.zeros((224, 224))
            mask = np.zeros((224, 224))
            mask[60:120,60:120]=1
            
        show_mask_image(image, mask, ax=axs[row, 2], title='benign_calc')
        
        if row < len(malig_mass_idx):
            k = malig_mass_idx[row]
            image = batch[0][k].numpy()[0]
            mask = batch[2][k].numpy()
        else:
            image = np.zeros((224, 224))
            mask = np.zeros((224, 224))
            mask[60:120,60:120]=1
            
        show_mask_image(image, mask, ax=axs[row, 3], title='malig_mass')
        
        if row < len(malig_calc_idx):
            k = malig_calc_idx[row]
            image = batch[0][k].numpy()[0]
            mask = batch[2][k].numpy()
        else:
            image = np.zeros((224, 224))
            mask = np.zeros((224, 224))
            mask[60:120,60:120]=1
            
        show_mask_image(image, mask, ax=axs[row, 4], title='malig_calc')
        
        fig.suptitle("mean = {:.2f}, std = {:.2f}".format(batch[0].mean(), batch[0].std()))
    
    return fig2img(fig)
            
        
def show_batch_ddsm(batch, num_rows=2):
    labels = batch[1].numpy()
    normal_idx = np.where(labels == 0)[0]
    cancer_idx = np.where(labels == 1)[0]

    fig, axs = plt.subplots(num_rows, 2, figsize=(10, 15))

    for row in range(num_rows):
        if row < len(normal_idx):
            k = normal_idx[row]
            image = batch[0][k].numpy()[0]
            mask = batch[2][k].numpy()
        else:
            image = np.zeros((224, 224))
            mask = np.zeros((224, 224))
            mask[60:120,60:120]=1
            
        show_mask_image(image, mask, ax=axs[row, 0], title='normal', multi_label=True)
        
        if row < len(cancer_idx):
            k = cancer_idx[row]
            image = batch[0][k].numpy()[0]
            mask = batch[2][k].numpy()
        else:
            image = np.zeros((224, 224))
            mask = np.zeros((224, 224))
            mask[60:120,60:120]=1
        
        show_mask_image(image, mask, ax=axs[row, 1], title='cancer', multi_label=True)
        
        fig.suptitle("mean = {:.2f}, std = {:.2f}".format(batch[0].mean(), batch[0].std()))
    
    return fig2img(fig)
                 

class VisualizeBatchPatchesCallback(Callback):
    def __init__(self, num_rows = 2):
        super().__init__()
        self.num_rows = num_rows

    def on_train_epoch_start(self, trainer, pl_module):
        # Get the first batch from the training dataloader
        train_dataloader = trainer.train_dataloader
        
        batch = next(iter(train_dataloader))
        #images, labels, masks = batch
        
        # Show the batch
        img = show_batch_patch(batch, num_rows=self.num_rows)
        experiment = trainer.logger.experiment
        experiment.log({"training batch": wandb.Image(img)})
        
        
        return super().on_train_epoch_start(trainer, pl_module)
        
        
    def on_validation_epoch_start(self, trainer, pl_module):
        # Get the first batch from the validation dataloader
        val_dataloader = trainer.val_dataloaders
        batch = next(iter(val_dataloader))
        #images, labels, masks = batch
        
        img = show_batch_patch(batch, num_rows=self.num_rows)
        experiment = trainer.logger.experiment
        experiment.log({"validation batch": wandb.Image(img)})
        
        
        
        return super().on_validation_epoch_start(trainer, pl_module)
    
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
        img = show_batch_ddsm(batch, num_rows=self.num_rows)
        experiment = trainer.logger.experiment
        experiment.log({"training batch": wandb.Image(img)})
        
        
        return super().on_train_epoch_start(trainer, pl_module)
        
        
    def on_validation_epoch_start(self, trainer, pl_module):
        # Get the first batch from the validation dataloader
        val_dataloader = trainer.val_dataloaders
        batch = next(iter(val_dataloader))
        #images, labels, masks = batch
        
        img = show_batch_ddsm(batch, num_rows=self.num_rows)
        experiment = trainer.logger.experiment
        experiment.log({"validation batch": wandb.Image(img)})
        
        
        
        return super().on_validation_epoch_start(trainer, pl_module)

 
class GradientNormLoggerCallback(pl.Callback):
    def __init__(self, log_freq: int = 10):
        """
        Args:
            log_freq: Frequency (in steps) to log the gradient norm.
        """
        self.log_freq = log_freq

    def log_grad_norms(self, trainer, grad_norms: dict):
        """Logs gradient norms to W&B or other loggers."""
        # Use trainer.global_step - 1 to avoid step conflicts
        if trainer.logger and hasattr(trainer.logger.experiment, "log"):
            trainer.logger.experiment.log(
                grad_norms, step=trainer.global_step + 1
            )

 
    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        """Log the clipped gradient norms before the optimizer step."""
        if trainer.global_step % self.log_freq == 0:
            grad_norms = {}
            total_norm = 0.0

            # Compute and log gradient norms after clipping
            for name, param in pl_module.named_parameters():
                if param.grad is not None:
                    grad_norm = torch.norm(param.grad).item()
                    total_norm += grad_norm ** 2
            
            grad_norms["total_grad_norm"] = total_norm ** 0.5

            # Log using the trainer's logger
            self.log_grad_norms(trainer, grad_norms)



class FreezePatchLayersCallback(Callback):
    def __init__(self, freeze_epochs):
        super().__init__()
        self.freeze_epochs = freeze_epochs

    def on_train_epoch_start(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        if current_epoch < self.freeze_epochs:
            pl_module.model.freeze_patch_layers()
            print(f"Epoch {current_epoch + 1}: Patch layers frozen")
        else:
            pl_module.model.unfreeze_patch_layers()
            print(f"Epoch {current_epoch + 1}: Patch layers unfrozen")