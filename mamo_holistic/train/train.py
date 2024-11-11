# Import necessary libraries
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets, models
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import argparse
import yaml

import sys

import pathlib
path = pathlib.Path(__file__).parent.parent.absolute()

sys.path.append(str(path))

from utils.load_config import load_config, get_parameter
from data.ddsm_dataset import DDSMPatchDataModule

# Define the Image Classification Model using PyTorch Lightning
class ImageClassifier(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        
          # Fetch hyperparameters from config
        self.num_classes = get_parameter(config, ["LightningModule", "num_classes"])
        self.optimizer = get_parameter(config, ["LightningModule", "optimizer"])
        self.learning_rate = get_parameter(config, ["LightningModule", "learning_rate"])
        self.optimizer_options = get_parameter(config, ["LightningModule", "optimizer_options"])

        # Save hyperparameters
        self.save_hyperparameters({
            'num_classes': self.num_classes,
            'optimizer': self.optimizer,
            'learning_rate': self.learning_rate,
            'optimizer_options': self.optimizer_options
        })
        
        
        # Define a pre-trained model (ResNet18 in this case)
        self.model = models.resnet18( weights=models.ResNet18_Weights.DEFAULT)
        # Modify the last layer to match the number of classes
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
        
        self.optimizer = self.create_optimizer()
        
        
        
    def create_optimizer(self):
        if self.optimizer.lower() == "adam":
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate, **self.optimizer_options)
        elif self.optimizer.lower() == "sgd":
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate, **self.optimizer_options)
        else:
            raise NotImplementedError(f"Unknown optimizer {self.optimizer}")
        

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

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
    args = parser.parse_args()

    # Load configuration from YAML file
    config = load_config(args.config_file)
   
    
    
    # Set up model, data module, logger, and checkpoint callback
    

    model = ImageClassifier(config=config)
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
        gpus=1 if torch.cuda.is_available() else None,
    )
    
    # Fit the model
    trainer.fit(model, data_module)
    
    # Test the model
    #trainer.test(model, data_module)
