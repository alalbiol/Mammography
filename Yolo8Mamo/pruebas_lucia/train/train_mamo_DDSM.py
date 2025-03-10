import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pathlib 
from torch.utils.data import random_split, TensorDataset, DataLoader
import torch.utils.data as data 
import sys
from sklearn.model_selection import train_test_split
import os 
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.patches as patches
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models import ResNet50_Weights
import lightning as L
from torchvision import transforms, datasets, models
import torchvision.transforms as T
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchmetrics
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import argparse
import yaml
#import wandb



sys.path.append("..")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from load_config import load_config, get_parameter
from classes_dataset_DDSM import DDSM_DataModule
from classes_model_DDSM import DDSM_CustomModel





# Training the model
if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train a ddsm detector.")
    parser.add_argument("--config_file", type=str, required=True, help="Path to the configuration file.")
    # parser.add_argument("--overrides", type=str, default = None, help="Overrides for the configuration file.")
   # parser.add_argument("--logger", type=str_to_bool, default=True, help="Use wandb for logging.")
    args = parser.parse_args()

    # Load configuration from YAML file
    config = load_config(args.config_file, #override_file=args.overrides
                         )
    
    from pprint import pprint
    pprint(config)
    
    GPU_TYPE = get_parameter(config, ["General", "gpu_type"],"None")
    if GPU_TYPE == "RTX 3090":
        torch.set_float32_matmul_precision('medium') # recomended for RTX 3090
   
    
    
    # Set up model, data module, logger, and checkpoint callback

    
    normalize = T.Compose([ T.ToTensor()])
    

    model = fasterrcnn_resnet50_fpn(num_classes=2, weights_backbone=ResNet50_Weights.IMAGENET1K_V1)
    data_module = DDSM_DataModule(config=config)
    
    
    #logger = get_logger(config) if args.logger else None


    # Trainer arguments
    #callbacks = create_callbacks(config)
    
    
    # Trainer
    #trainer_kwargs = get_parameter(config, ["Trainer"], mode="default", default={})
    
    #print("Trainer kwargs: ", trainer_kwargs)
    
    Lmodel = DDSM_CustomModel(model)


    checkpoint_callback = ModelCheckpoint( # Para que guarde las tres mejores épocas 
        monitor='val_map',
        dirpath='/home/lloprib/',
        filename='mamo-{epoch:02d}-{val_map:.2f}',
        save_top_k=3,
        mode='max',
    )

    trainer = L.Trainer(
        max_epochs=1, 
        devices='auto',
        accelerator='gpu',
        default_root_dir='/home/lloprib/proyecto_mam/Mammography/Yolo8Mamo/pruebas_lucia/checkpoints/',
        callbacks=[checkpoint_callback]
    )

    trainer.fit(Lmodel, data_module)


    print("Training finished")
    

