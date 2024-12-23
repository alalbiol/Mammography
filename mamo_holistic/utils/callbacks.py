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
            pl_module.model.freeze_patch_model()
            print(f"Epoch {current_epoch}: Patch layers frozen")
        else:
            pl_module.model.unfreeze_patch_model()
            print(f"Epoch {current_epoch}: Patch layers unfrozen")

class FreezeSwinLayersCallback(Callback):
    def __init__(self, freeze_epochs=2):
        super().__init__()
        self.freeze_epochs = freeze_epochs

    def on_train_epoch_start(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        if current_epoch < self.freeze_epochs:
            
            swin_model = pl_module.model
            for param in swin_model.parameters():
                param.requires_grad = False
            
            for param in swin_model.head.fc.parameters():
                param.requires_grad = True   
            print(f"Epoch {current_epoch}: Swin layers frozen except the last layer")
        else:
            swin_model = pl_module.model
            for param in swin_model.parameters():
                param.requires_grad = True
            print(f"Epoch {current_epoch}: Patch layers unfrozen")


            
class EMACallback(Callback):
    def __init__(self, decay=0.999):
        """
        Initializes the EMA callback.
        Args:
            decay (float): Decay rate for the moving average. Values closer to 1 retain weights longer.
        """
        super().__init__()
        self.decay = decay
        self.ema_weights = {}

    def setup(self, trainer, pl_module, stage=None):
        """
        Initializes EMA weights during setup. These remain on CPU initially.
        """
        self.ema_weights = {
            name: param.clone().detach()
            for name, param in pl_module.named_parameters() if param.requires_grad
        }

    def on_train_start(self, trainer, pl_module):
        """
        Moves EMA weights to the same device as the model at the start of training.
        """
        device = next(pl_module.parameters()).device
        for name, ema_param in self.ema_weights.items():
            self.ema_weights[name] = ema_param.to(device)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """
        Updates the EMA weights at the end of each training batch.
        """
        for name, param in pl_module.named_parameters():
            if param.requires_grad:
                self.ema_weights[name].data = (
                    self.decay * self.ema_weights[name].data + (1 - self.decay) * param.data
                )

    def on_validation_epoch_start(self, trainer, pl_module):
        """
        Replaces the model's weights with EMA weights before validation.
        """
        print("Replacing model weights with EMA weights for validation")
        self._backup_and_apply_ema(pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Restores the original model weights after validation.
        """
        print("Restoring original model weights after validation")
        self._restore_original_weights(pl_module)

    def _backup_and_apply_ema(self, pl_module):
        """
        Backs up the model's weights and applies the EMA weights.
        """
        self.backup_weights = {}
        for name, param in pl_module.named_parameters():
            if param.requires_grad:
                self.backup_weights[name] = param.clone().detach()
                param.data.copy_(self.ema_weights[name].data)

    def _restore_original_weights(self, pl_module):
        """
        Restores the original model weights from the backup.
        """
        for name, param in pl_module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup_weights[name].data)

    def state_dict(self):
        """
        Saves the EMA weights for checkpointing.
        """
        return {"ema_weights": self.ema_weights}

    def load_state_dict(self, state_dict):
        """
        Loads the EMA weights from a checkpoint.
        """
        self.ema_weights = state_dict["ema_weights"]



from torch import nn
from peft import get_peft_model, LoraConfig, TaskType
task_type=TaskType.FEATURE_EXTRACTION



class LoRACallback(Callback):
    def __init__(self, task_type="FEATURE_EXTRACTION", r=8, lora_alpha=32, lora_dropout=0.1, target_modules=["qkv"]):
        """
        Args:
            task_type (str): Type of task. Must be one of the supported types.
            r (int): Rank of the LoRA updates.
            lora_alpha (int): Scaling factor for LoRA updates.
            lora_dropout (float): Dropout rate applied to LoRA.
            target_modules (list[str]): List of module names in the model to which LoRA should be applied.
        """
        super().__init__()
        task_type=TaskType.FEATURE_EXTRACTION

        self.lora_config = LoraConfig(
            task_type=task_type,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules or ["q", "k", "v"],  # Default for transformer-like layers
        )

    def setup(self, trainer, pl_module, stage=None):
        """
        Called during trainer setup.
        """
        if stage == "fit" or stage is None:
            print("Applying LoRA to the model...")
            # Wrap the model with LoRA
            pl_module.model = get_peft_model(pl_module.model, self.lora_config)

    def on_train_start(self, trainer, pl_module):
        """
        Called at the start of training.
        """
        
        trainable_params = sum(p.numel() for p in pl_module.model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters after LoRA: {trainable_params}")
        
        

class ReduceLROnEpochsCallback(pl.Callback):
    def __init__(self, factor=0.1, every_n_epochs=10, min_lr=1e-6):
        """
        Callback to reduce the learning rate by a factor every n epochs.

        Args:
            factor (float): The factor by which to reduce the learning rate (e.g., 0.1 for 10% of the original).
            every_n_epochs (int): Reduce the learning rate every this many epochs.
        """
        super().__init__()
        self.factor = factor
        self.every_n_epochs = every_n_epochs
        self.min_lr = min_lr
        
 

    def on_validation_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch

        if (current_epoch + 1) % self.every_n_epochs == 0:  # Adjust LR every `every_n_epochs`
            for optimizer in trainer.optimizers:
                for param_group in optimizer.param_groups:
                    old_lr = param_group['lr']
                    
                    if old_lr <= self.min_lr:
                        print(f"LR already at minimum value {self.min_lr:.6f} for optimizer {optimizer}")
                        continue
                    
                    new_lr = old_lr * self.factor
                    if new_lr < self.min_lr:
                        new_lr = self.min_lr
                       
                    param_group['lr'] = new_lr            
                    print(f"Reduced LR from {old_lr:.6f} to {new_lr:.6f}")
