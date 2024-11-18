import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

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
