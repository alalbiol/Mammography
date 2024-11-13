# Import necessary libraries
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets, models
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import argparse
import yaml
import wandb
import matplotlib.pyplot as plt
import numpy as np

import sys

import pathlib
path = pathlib.Path(__file__).parent.parent.absolute()

sys.path.append(str(path))

from utils.load_config import load_config, get_parameter
from data.ddsm_dataset import DDSMPatchDataModule
from models.model_selector import get_patch_model

# Define the Image Classification Model using PyTorch Lightning
class DDSMPatchClassifier(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        
          # Fetch hyperparameters from config
        self.num_classes = get_parameter(config, ["LightningModule", "num_classes"])
        self.optimizer_type = get_parameter(config, ["LightningModule", "optimizer_type"])
        self.learning_rate = get_parameter(config, ["LightningModule", "learning_rate"])
        self.optimizer_options = get_parameter(config, ["LightningModule", "optimizer_options"], default={})
        self.lr_scheduler = get_parameter(config, ["LightningModule", "lr_scheduler"], default=None)
        self.lr_scheduler_options = get_parameter(config, ["LightningModule", "lr_scheduler_options"], default={})
        self.n_hard_mining = get_parameter(config, ["LightningModule", "n_hard_mining"], default=-1)

        assert isinstance(self.lr_scheduler_options['min_lr'],float), "min_lr must be a float"

        # Save hyperparameters
        self.save_hyperparameters({
            'num_classes': self.num_classes,
            'optimizer': self.optimizer_type,
            'learning_rate': self.learning_rate,
            'optimizer_options': self.optimizer_options
        })
        
        # Define a pre-trained model (ResNet18 in this case)
        self.model = get_patch_model(get_parameter(config, ["LightningModule", "model_name"]), num_classes=self.num_classes)
        
        # Metrics initialization
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes, average='macro')
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes, average='macro')
        self.val_confusion_matrix = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.num_classes)
        
        self.val_AUROC_5 = torchmetrics.AUROC(task = "multiclass", num_classes=self.num_classes, average='macro')
        self.val_AUROC_4 = torchmetrics.AUROC(task = "multiclass", num_classes=self.num_classes-1, average='macro')

        
        self.tp = torch.zeros(self.num_classes)
        self.fp = torch.zeros(self.num_classes)
        self.fn = torch.zeros(self.num_classes)
        self.total = torch.zeros(self.num_classes) 
        
        # self.val_precision = torchmetrics.Precision(num_classes=num_classes, average='macro')
        # self.val_recall = torchmetrics.Recall(num_classes=num_classes, average='macro')
        # self.val_f1 = torchmetrics.F1Score(num_classes=num_classes, average='macro')
        
        class_names = ['NORMAL',
            'MASS_BENIGN',
            'CALCIFICATION_BENIGN',
            'MASS_MALIGNANT',
            'CALCIFICATION_MALIGNANT',
        ]
        self.class_to_idx = {class_name: i for i, class_name in enumerate(class_names)}
        self.idx_to_class = {i: class_name for i, class_name in enumerate(class_names)}

       
        
    def configure_optimizers(self): # called by the trainer
        if self.optimizer_type.lower() == "adam":
            optimizer =  torch.optim.Adam(self.parameters(), lr=self.learning_rate, **self.optimizer_options)
        elif self.optimizer_type.lower() == "sgd":
            optimizer =  torch.optim.SGD(self.parameters(), lr=self.learning_rate, **self.optimizer_options)
        else:
            raise NotImplementedError(f"Unknown optimizer {self.optimizer}")
        
        
        if self.lr_scheduler is not None:
            print("Using LR Scheduler")
            if self.lr_scheduler == "ReduceLROnPlateau":            
                # Set up ReduceLROnPlateau scheduler
                scheduler = {
                    'scheduler': ReduceLROnPlateau(optimizer, **self.lr_scheduler_options),
                    'monitor': 'val_loss',  # Monitors validation loss
                    'interval': 'epoch',
                    'frequency': 1
                }
        else:
            return optimizer            
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
        


    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]
        
        logits = self(x)
        
        preds = torch.argmax(logits, dim=1)
                
        acc = self.train_accuracy(preds, y)
        
        if self.n_hard_mining > 0:
            #split logits into two parts. First those with y==0 and second those with y!=0
            #sort the first part and take the first n_hard_mining acorrding to the logit value
            logits_bg = logits[y==0]
            logits_fg = logits[y!=0]
            y_bg = y[y==0]
            y_fg = y[y!=0]
            
            # Get the top k values and their indices from the first column
            top_k_values, top_k_indices = torch.topk(logits_bg[:, 0], self.n_hard_mining)

            # Select the top k rows
            top_logits_bg = logits_bg[top_k_indices]
            top_labels_bg = y_bg[top_k_indices]
            
            logits = torch.cat((top_logits_bg, logits_fg), dim=0)
            y = torch.cat((top_labels_bg, y_fg), dim=0)
            
        
        loss = F.cross_entropy(logits, y)

        self.log("train_acc", acc, prog_bar=True)       
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]
        
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        
        self.val_confusion_matrix(preds, y)

        # Update confusion matrix components
        for i in range(self.num_classes):
            self.tp[i] += torch.sum((preds == i) & (y == i)).item()  # True positives
            self.fp[i] += torch.sum((preds == i) & (y != i)).item()  # False positives
            self.fn[i] += torch.sum((preds != i) & (y == i)).item()  # False negatives
            self.total[i] += torch.sum(y == i).item()  # Total instances of class i

        # Log the loss (optional)
        self.log("val/loss", loss, prog_bar=True)
        
        
        acc = (preds == y).float().mean() # global accuracy
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        
        # probs = F.softmax(logits, dim=1)
        # self.val_AUROC_5(probs, y)
        # self.log("val_AUROC_5", self.val_AUROC_5)
        
        # print("probs",probs.shape)
        # probs_4 = probs[y>0,1:]
        # labels_4 = y[y>0]-1
        # print("probs_4",probs_4.shape)
        # print("labels_4",labels_4.shape)
                
        # self.val_AUROC_4(probs_4, labels_4)
        # self.log("val_AUROC_4", self.val_AUROC_4)
        # import sys
        # sys.exit()
        
        return loss
    
    # def test_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self(x)
    #     loss = F.cross_entropy(y_hat, y)
    #     acc = (y_hat.argmax(dim=1) == y).float().mean()
    #     self.log("test_loss", loss)
    #     self.log("test_acc", acc)
    #     return loss
    
    def on_training_epoch_end(self, outputs):
        # log train accuracy
        self.log("train_acc_epoch", self.train_accuracy.compute(), prog_bar=True)
        self.train_accuracy.reset()
        
    
    def on_validation_epoch_end(self):
        # Compute precision, recall, and f1 for each class
        precision_per_class = self.tp / (self.tp + self.fp + 1e-10)  # Add small epsilon to avoid division by zero
        recall_per_class = self.tp / (self.tp + self.fn + 1e-10)
        f1_per_class = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class + 1e-10)

        # Log per-class metrics at epoch end
        for i in range(1,self.num_classes):
            class_name = self.idx_to_class[i]
            self.log(f"val/precision_class_{class_name}", precision_per_class[i])
            self.log(f"val/recall_class_{class_name}", recall_per_class[i])
            self.log(f"val/f1_class_{class_name}", f1_per_class[i])

        # Calculate and log macro average
        macro_precision = precision_per_class[1:].mean()
        macro_recall = recall_per_class[1:].mean()
        macro_f1 = f1_per_class[1:].mean()

        self.log("val/macro_precision", macro_precision, prog_bar=True)
        self.log("val/macro_recall", macro_recall, prog_bar=True)
        self.log("val/macro_f1", macro_f1, prog_bar=True)

        # Reset metrics to avoid accumulation over multiple epochs
        self.tp.zero_()
        self.fp.zero_()
        self.fn.zero_()
        self.total.zero_()
        
        self.val_confusion_matrix.compute()
        fig, ax = self.val_confusion_matrix.plot()
                
        experiment = self.logger.experiment
        experiment.log({"val confusion_matrix": wandb.Image(fig)})
        self.val_confusion_matrix.reset()
        
        # val_auroc_5 = self.val_AUROC_5.compute()
        # val_auroc_4 = self.val_AUROC_4.compute()
        # self.log("val_AUROC_5", val_auroc_5)
        # self.log("val_AUROC_4", val_auroc_4)
        # self.val_AUROC_5.reset()
        # self.val_AUROC_4.reset()
    
    

def get_logger(config):
    # Check if the logger is set to WandB
    if get_parameter(config, ["Logger", "type"]) == "wandb":
        # Retrieve the WandB API key
        options = {}
        options['project'] = get_parameter(config, ["Logger", "project"])
        options['name'] = get_parameter(config, ["Logger", "name"])
        options['save_dir'] = get_parameter(config, ["Logger", "save_dir"])
        # Initialize the WandB logger
        logger = WandbLogger(**options)
    else:
        logger = None
    return logger


def create_callbacks(config):
    callbacks = []
    
    callbacks_dict = get_parameter(config, ["Callbacks"])
    
    for callback_name in callbacks_dict:
        if callback_name == "EarlyStopping":
            from pytorch_lightning.callbacks import EarlyStopping
            callbacks.append(EarlyStopping(**callbacks_dict[callback_name]))
        elif callback_name == "ModelCheckpoint":
            from pytorch_lightning.callbacks import ModelCheckpoint
            callbacks.append(ModelCheckpoint(**callbacks_dict[callback_name]))
        elif callback_name == "LearningRateMonitor":
            from pytorch_lightning.callbacks import LearningRateMonitor
            callbacks.append(LearningRateMonitor(**callbacks_dict[callback_name]))
        else:
            raise NotImplementedError(f"Unknown callback {callback_name}")
    return callbacks



# Training the model
if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train a ddsm patch classifier.")
    parser.add_argument("--config_file", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--overrides", type=str, default = None, help="Overrides for the configuration file.")
    args = parser.parse_args()

    # Load configuration from YAML file
    config = load_config(args.config_file, override_file=args.overrides)
    
    from pprint import pprint
    pprint(config)
    
    GPU_TYPE = get_parameter(config, ["General", "gpu_type"],"None")
    if GPU_TYPE == "RTX 3090":
        torch.set_float32_matmul_precision('medium') # recomended for RTX 3090
   
    
    
    # Set up model, data module, logger, and checkpoint callback
    

    model = DDSMPatchClassifier(config=config)
    data_module = DDSMPatchDataModule(config=config)
    logger = get_logger(config)


    # Trainer arguments
    max_epochs = get_parameter(config, ["Trainer", "max_epochs"], mode="warn")
    callbacks = create_callbacks(config)
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=logger,
        callbacks= callbacks,
        accelerator = 'gpu' if torch.cuda.is_available() else "cpu",
    )
    
    # Fit the model
    trainer.fit(model, data_module)
    
    # Test the model
    #trainer.test(model, data_module)
