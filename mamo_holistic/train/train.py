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
from data.ddsm_dataset import DDSMPatchDataModule
from models.model_selector import get_patch_model
from losses import get_loss
from utils.traininig import mixup_data

from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve


# Define the Image Classification Model using PyTorch Lightning
class DDSMPatchClassifier(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        
          # Fetch hyperparameters from config
        self.model_name = get_parameter(config, ["LightningModule", "model_name"])
        self.num_classes = get_parameter(config, ["LightningModule", "num_classes"])
        self.optimizer_type = get_parameter(config, ["LightningModule", "optimizer_type"])
        self.learning_rate = get_parameter(config, ["LightningModule", "learning_rate"])
        self.optimizer_options = get_parameter(config, ["LightningModule", "optimizer_options"], default={})
        self.lr_scheduler = get_parameter(config, ["LightningModule", "lr_scheduler"], default=None)
        self.lr_scheduler_options = get_parameter(config, ["LightningModule", "lr_scheduler_options"], default={})
        self.n_hard_mining = get_parameter(config, ["LightningModule", "n_hard_mining"], default=-1)
        self.loss_name = get_parameter(config, ["LightningModule", "loss_name"], default="cross_entropy")
        self.loss_params = get_parameter(config, ["LightningModule", "loss_params"], default={})
        self.mixup_alpha = get_parameter(config, ["LightningModule", "mixup_alpha"], default=-1)
        self.pretrained_weights = get_parameter(config, ["LightningModule", "pretrained_weights"], default=None)

        if 'min_lr' in self.lr_scheduler_options:
            assert isinstance(self.lr_scheduler_options['min_lr'],float), "min_lr must be a float"

        # Save hyperparameters
        self.save_hyperparameters({
            'num_classes': self.num_classes,
            'optimizer': self.optimizer_type,
            'learning_rate': self.learning_rate,
            'optimizer_options': self.optimizer_options,
            'lr_scheduler': self.lr_scheduler,
            'lr_scheduler_options': self.lr_scheduler_options,
            'n_hard_mining': self.n_hard_mining,
            'loss_name': self.loss_name,
            'loss_params': self.loss_params,
            'mixup_alpha': self.mixup_alpha,
        })
        
        # Define a pre-trained model (ResNet18 in this case)
        self.model = get_patch_model(self.model_name, num_classes=self.num_classes)
        
        self.loss_fn = get_loss(self.loss_name, **self.loss_params)
        
        # Metrics initialization
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes, average='macro')
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes, average='macro')
        self.val_confusion_matrix = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.num_classes)
        self.train_confusion_matrix = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.num_classes)
        
        
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
        
        if self.pretrained_weights is not None:
            print("Loading pretrained weights from: ", self.pretrained_weights)
            self.model.load_weights_from_patch_model(self.pretrained_weights)


        
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
                    'monitor': 'val_auroc',  # Monitors validation loss
                    'interval': 'epoch',
                    'frequency': 1
                }
        else:
            return optimizer            
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
        


    def forward(self, x):
        return self.model(x)
    
    def on_train_epoch_start(self):
        self.train_outputs = torch.empty(0, self.num_classes, device='cpu')
        self.train_targets = torch.empty(0, dtype=torch.long, device='cpu')
    
    def training_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]
        #mask = batch[2]

    
        #check if mixup is enabled
        if  self.mixup_alpha > 0:
            x, y_a, y_b, lam = mixup_data(x, y, self.mixup_alpha)
            y = y_a if lam > 0.5 else y_b
            logits = self(x)
            loss = lam * self.loss_fn(logits, y_a) + (1 - lam) * self.loss_fn(logits, y_b)
        else:        
            logits = self(x)
            loss = self.loss_fn(logits, y)
        
        preds = torch.argmax(logits, dim=1)
                
        acc = self.train_accuracy(preds, y)
        
        # if self.n_hard_mining > 0:
        #     #split logits into two parts. First those with y==0 and second those with y!=0
        #     #sort the first part and take the first n_hard_mining acorrding to the logit value
        #     logits_bg = logits[y==0]
        #     logits_fg = logits[y!=0]
        #     y_bg = y[y==0]
        #     y_fg = y[y!=0]
            
        #     # Get the top k values and their indices from the first column
        #     top_k_values, top_k_indices = torch.topk(logits_bg[:, 0], self.n_hard_mining)

        #     # Select the top k rows
        #     top_logits_bg = logits_bg[top_k_indices]
        #     top_labels_bg = y_bg[top_k_indices]
            
        #     logits = torch.cat((top_logits_bg, logits_fg), dim=0)
        #     y = torch.cat((top_labels_bg, y_fg), dim=0)
            
        
        preds = torch.argmax(logits, dim=1)
        self.train_confusion_matrix(preds, y)
        
        self.log("train_acc", acc, prog_bar=True)       
        self.log("train_loss", loss)
        
        # Store outputs and targets for AUROC and PRROC calculation
        self.train_outputs = torch.cat((self.train_outputs, logits.detach().cpu()), dim=0)
        self.train_targets = torch.cat((self.train_targets, y.detach().cpu()), dim=0)
    
        return loss
    
    def on_validation_epoch_start(self):
        self.val_outputs = torch.empty(0, self.num_classes, device='cpu')
        self.val_targets = torch.empty(0, dtype=torch.long, device='cpu')
    
    def validation_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]
        #mask = batch[2]
        
        logits = self(x)
        loss = self.loss_fn(logits, y)
        
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
        
        # Store outputs and targets for AUROC and PRROC calculation
        self.val_outputs = torch.cat((self.val_outputs, logits.detach().cpu()), dim=0)
        self.val_targets = torch.cat((self.val_targets, y.detach().cpu()), dim=0) 
                  
        return loss
    
    # def test_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self(x)
    #     loss = F.cross_entropy(y_hat, y)
    #     acc = (y_hat.argmax(dim=1) == y).float().mean()
    #     self.log("test_loss", loss)
    #     self.log("test_acc", acc)
    #     return loss
    
    # compute training metrics at the end of training epoch
    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        self.train_confusion_matrix.compute()
        fig, ax = self.train_confusion_matrix.plot()
        
        if isinstance(self.trainer.logger,WandbLogger):                
            experiment = self.logger.experiment
            experiment.log({"train/train confusion_matrix": wandb.Image(fig2img(fig))})
        self.train_confusion_matrix.reset()
        
        self.log("train/train_acc_epoch", self.train_accuracy.compute(), prog_bar=True)
        self.train_accuracy.reset()
        
           # Calculate cancer probability
        if self.num_classes == 5:
            cancer_prob = self.train_outputs[:, 3] + self.train_outputs[:, 4]
            cancer_label = (self.train_targets > 2).long()
        else: # binary labels
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
            experiment.log({"train/train ROC curve": wandb.Image(fig2img(fig_roc))})
            experiment.log({"train/train PR curve": wandb.Image(fig2img(fig_pr))})

            
    
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
        if isinstance( self.trainer.logger, WandbLogger):     
            experiment = self.logger.experiment
            experiment.log({"val/val confusion_matrix": wandb.Image(fig2img(fig))})
        self.val_confusion_matrix.reset()
        
        
           # Calculate cancer probability
        if self.num_classes == 5:
            cancer_prob = self.val_outputs[:, 3] + self.val_outputs[:, 4]
            cancer_label = (self.val_targets > 2).long()
        else: # binary labels
            cancer_prob = self.val_outputs[:, 1]
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
            
        # val_auroc_5 = self.val_AUROC_5.compute()
        # val_auroc_4 = self.val_AUROC_4.compute()
        # self.log("val_AUROC_5", val_auroc_5)
        # self.log("val_AUROC_4", val_auroc_4)
        # self.val_AUROC_5.reset()
        # self.val_AUROC_4.reset()
                    
    # def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
    #     super().configure_gradient_clipping(optimizer, gradient_clip_val, gradient_clip_algorithm)
        
    #     trainer = self.trainer
    #     pl_module = self
    #     if trainer.global_step % trainer.log_every_n_steps == 0:
    #         total_norm = 0.0
    #         for name, param in pl_module.named_parameters():
    #             if param.grad is not None:
    #                 grad_norm = torch.norm(param.grad).item()
    #                 total_norm += grad_norm ** 2  # Accumulate squared norms
            
            
    #         if trainer.logger and hasattr(trainer.logger.experiment, "log"):
    #             trainer.logger.experiment.log(
    #                 {"total_grad_norm_after_clipping": total_norm ** 0.5}, step=trainer.global_step+1)
                
              

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
        elif callback_name == "VisualizeBatchPatches":
            from utils.callbacks import VisualizeBatchPatchesCallback
            callbacks.append(VisualizeBatchPatchesCallback(**callbacks_dict[callback_name]))
        elif callback_name == "GradientNormLogger":
            from utils.callbacks import GradientNormLoggerCallback
            params = callbacks_dict[callback_name] if callbacks_dict[callback_name] is not None else {}
            callbacks.append(GradientNormLoggerCallback(**params))
        elif callback_name == "EMACallback":
            from utils.callbacks import EMACallback
            callbacks.append(EMACallback(**callbacks_dict[callback_name]))

        else:
            raise NotImplementedError(f"Unknown callback {callback_name}")
    return callbacks



# Training the model
if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train a ddsm patch classifier.")
    parser.add_argument("--config_file", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--overrides", type=str, default = None, help="Overrides for the configuration file.")
    parser.add_argument("--logger", type=str_to_bool, default=True, help="Use wandb for logging.")
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
    
    
    logger = get_logger(config) if args.logger else None


    # Trainer arguments
    callbacks = create_callbacks(config)
    
    
    # Trainer
    trainer_kwargs = get_parameter(config, ["Trainer"], mode="default", default={})
    
    print("Trainer kwargs: ", trainer_kwargs)
    
    trainer = pl.Trainer(
        logger=logger,
        callbacks= callbacks,
        accelerator = 'gpu' if torch.cuda.is_available() else "cpu",
        **trainer_kwargs
    )
    
    # Fit the model
    trainer.fit(model, data_module)
    
    # Test the model
    #trainer.test(model, data_module)
