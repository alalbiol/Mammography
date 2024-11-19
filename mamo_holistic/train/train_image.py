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
from utils.utils import fig2img, str_to_bool
from utils.utils import plot_roc_curve, plot_pr_curve
from data.ddsm_dataset import DDSMImageDataModule
from models.model_selector import get_image_model
from losses import get_loss
from utils.traininig import mixup_data

from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve


# Define the Image Classification Model using PyTorch Lightning
class DDSMImageClassifier(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        
          # Fetch hyperparameters from config
        self.model_name = get_parameter(config, ["LightningModule", "model_name"])
        self.num_classes = get_parameter(config, ["LightningModule", "num_classes"], default=2)
        self.optimizer_type = get_parameter(config, ["LightningModule", "optimizer_type"], default="adam")
        self.learning_rate = get_parameter(config, ["LightningModule", "learning_rate"], default=1e-3)
        self.optimizer_options = get_parameter(config, ["LightningModule", "optimizer_options"], default={})
        self.lr_scheduler = get_parameter(config, ["LightningModule", "lr_scheduler"], default=None)
        self.lr_scheduler_options = get_parameter(config, ["LightningModule", "lr_scheduler_options"], default={})
        self.loss_name = get_parameter(config, ["LightningModule", "loss_name"], default="cross_entropy")
        self.loss_params = get_parameter(config, ["LightningModule", "loss_params"], default={})
        self.mixup_alpha = get_parameter(config, ["LightningModule", "mixup_alpha"], default=-1)
        self.model_name = get_parameter(config, ["LightningModule", "model_name"])
        self.model_params = get_parameter(config, ["LightningModule", "model_params"], default={})

    

        # Save hyperparameters
        self.save_hyperparameters({
            'model_name': self.model_name,
            'model_params': self.model_params,
            'num_classes': self.num_classes,
            'optimizer': self.optimizer_type,
            'learning_rate': self.learning_rate,
            'optimizer_options': self.optimizer_options,
            'lr_scheduler': self.lr_scheduler,
            'lr_scheduler_options': self.lr_scheduler_options,
            'loss_name': self.loss_name,
            'loss_params': self.loss_params,
            'mixup_alpha': self.mixup_alpha,
        })
        
        # Define a pre-trained model (ResNet18 in this case)
        self.model = get_image_model(self.model_name, num_classes=self.num_classes, **self.model_params)
        
        self.loss_fn = get_loss(self.loss_name, **self.loss_params)
        
        # Metrics initialization
        self.train_accuracy = torchmetrics.Accuracy('multiclass', num_classes=self.num_classes)
        self.val_accuracy = torchmetrics.Accuracy('multiclass', num_classes=self.num_classes)
        self.train_confusion_matrix = torchmetrics.ConfusionMatrix('multiclass', num_classes= self.num_classes)
        self.val_confusion_matrix = torchmetrics.ConfusionMatrix('multiclass', num_classes= self.num_classes)
        
        
        class_names = ['NORMAL',
            'CANCER',
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
    
    def on_train_epoch_start(self):
        self.train_outputs = torch.empty(0,  device='cpu')
        self.train_targets = torch.empty(0, dtype=torch.long, device='cpu')

        return super().on_train_epoch_start()
        

    def training_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]
        
        
        logits = self(x)
        
        preds = torch.argmax(logits, dim=1)
                
        acc = self.train_accuracy(preds, y)
        self.log("train_acc", acc, prog_bar=True)       
                
        preds = torch.argmax(logits, dim=1)
        self.train_confusion_matrix(preds, y)
        
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)
        
        
        probs = F.softmax(logits, dim=1)
        
        # Store outputs and targets for AUROC and PRROC calculation
        

        self.train_outputs = torch.cat((self.train_outputs, probs[:,1].detach().cpu()), dim=0)
        self.train_targets = torch.cat((self.train_targets, y.detach().cpu()), dim=0)
    
        return loss
    
    def on_validation_epoch_start(self):
        self.val_outputs = torch.empty(0,  device='cpu')
        self.val_targets = torch.empty(0, dtype=torch.long, device='cpu')
        
    def on_test_epoch_start(self):
        self.on_validation_epoch_start()
    
    def validation_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]
        
        logits = self(x)
        loss = self.loss_fn(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        
        self.val_confusion_matrix(preds, y)

        # Log the loss (optional)
        self.log("val/loss", loss, prog_bar=True)
        acc = (preds == y).float().mean() # global accuracy
        
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        
        probs = F.softmax(logits, dim=1)
        
        # Store outputs and targets for AUROC and PRROC calculation
        #print("probs", probs[:,1].shape)
        #print("self.val_outputs", self.val_outputs.shape)
        
        self.val_outputs = torch.cat((self.val_outputs, probs[:,1].detach().cpu()), dim=0)
        self.val_targets = torch.cat((self.val_targets, y.detach().cpu()), dim=0) 

        return loss
    
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    # compute training metrics at the end of training epoch
    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        self.train_confusion_matrix.compute()
        fig, ax = self.train_confusion_matrix.plot()
        
        if isinstance(self.trainer.logger,WandbLogger):                
            experiment = self.logger.experiment
            experiment.log({"train confusion_matrix": wandb.Image(fig2img(fig))})
        self.train_confusion_matrix.reset()
        
        self.log("train_acc_epoch", self.train_accuracy.compute(), prog_bar=True)
        self.train_accuracy.reset()
        
        # Calculate cancer probability
       
        cancer_prob = self.train_outputs[:, 1]
        cancer_label = self.train_targets
        
        # Compute AUROC and PRROC
        auroc = roc_auc_score(cancer_label, cancer_prob)
        precision, recall, _ = precision_recall_curve(cancer_label, cancer_prob)
        prroc = average_precision_score(cancer_label, cancer_prob)
        
        # Log AUROC and PRROC
        self.log("train_auroc", auroc, prog_bar=True)
        self.log("train_prroc", prroc, prog_bar=True)
        
        # Plot and log ROC and PR curves
        fpr, tpr, _ = roc_curve(cancer_label, cancer_prob)
        fig_roc, ax_roc = plot_roc_curve(fpr, tpr)
        fig_pr, ax_pr = plot_pr_curve(precision, recall)
        
        if isinstance(self.trainer.logger, WandbLogger):
            experiment.log({"train ROC curve": wandb.Image(fig2img(fig_roc))})
            experiment.log({"train PR curve": wandb.Image(fig2img(fig_pr))})

            
    
    def on_validation_epoch_end(self):

        
        self.val_confusion_matrix.compute()
        fig, ax = self.val_confusion_matrix.plot() 
        if isinstance( self.trainer.logger, WandbLogger):     
            experiment = self.logger.experiment
            experiment.log({"val confusion_matrix": wandb.Image(fig2img(fig))})
        self.val_confusion_matrix.reset()
        
        
        # Calculate cancer probability

        cancer_prob = self.val_outputs
        cancer_label = self.val_targets
    
        # for the first epoch, skip if all labels are the same
        if cancer_label.sum() == 0 or cancer_label.sum() == len(cancer_label): # Skip if all labels are the same
            return
        
        # Compute AUROC and PRROC
        auroc = roc_auc_score(cancer_label, cancer_prob)
        precision, recall, _ = precision_recall_curve(cancer_label, cancer_prob)
        prroc = average_precision_score(cancer_label, cancer_prob)
        
        # Log AUROC and PRROC
        self.log("val_auroc", auroc, prog_bar=True)
        self.log("val_prroc", prroc, prog_bar=True)
        
        # Plot and log ROC and PR curves
        fpr, tpr, _ = roc_curve(cancer_label, cancer_prob)
        fig_roc, ax_roc = plot_roc_curve(fpr, tpr)
        fig_pr, ax_pr = plot_pr_curve(precision, recall)
        
        if isinstance(self.trainer.logger, WandbLogger):
            experiment = self.logger.experiment
            experiment.log({"val ROC curve": wandb.Image(fig2img(fig_roc))})
            experiment.log({"val PR curve": wandb.Image(fig2img(fig_pr))})
            
   
    def on_test_epoch_end(self):
        self.on_validation_epoch_end()

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
        elif callback_name == "LearningRateWarmUp":
            from utils.callbacks import LearningRateWarmUpCallback
            callbacks.append(LearningRateWarmUpCallback(**callbacks_dict[callback_name]))
        else:
            raise NotImplementedError(f"Unknown callback {callback_name}")
    return callbacks



# Training the model
if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train a ddsm patch classifier.")
    parser.add_argument("--task", type=str, default="train", help="The task to perform.")
    parser.add_argument("--config_file", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--overrides", type=str, default = None, help="Overrides for the configuration file.")
    parser.add_argument("--logger", type=str_to_bool, default=True, help="Use wandb for logging.")
    args = parser.parse_args()

    # Load configuration from YAML file
    config = load_config(args.config_file, override_file=args.overrides)
    
    
    
    GPU_TYPE = get_parameter(config, ["General", "gpu_type"],"None")
    if GPU_TYPE == "RTX 3090":
        torch.set_float32_matmul_precision('medium') # recomended for RTX 3090
   
    
    
    # Set up model, data module, logger, and checkpoint callback
    

    model = DDSMImageClassifier(config=config)
    data_module = DDSMImageDataModule(config=config)
    
    
    logger = get_logger(config) if args.logger else None


    # Trainer arguments
    callbacks = create_callbacks(config)
    
    
    # Trainer
    trainer_kwargs = get_parameter(config, ["Trainer"], mode="default", default={})
    trainer = pl.Trainer(
        logger=logger,
        callbacks= callbacks,
        accelerator = 'gpu' if torch.cuda.is_available() else "cpu",
        **trainer_kwargs
    )
    
    # Fit the model
    if args.task == "train":
        trainer.fit(model, data_module)
    elif args.task == "validate":
        trainer.validate(model, data_module)
    elif args.task == "test":
        trainer.test(model, data_module)    
    else:
        raise ValueError(f"Unknown task {args.task}")
    
    # Test the model
    #trainer.test(model, data_module)
