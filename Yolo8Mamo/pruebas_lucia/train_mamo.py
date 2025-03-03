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
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.patches as patches
from clases_train_mamo import bounding_boxes_coord, CustomDataset, CustomModel, DDSMPatchDataModule
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models import ResNet50_Weights
import lightning as L
from torchvision import transforms, datasets, models
import torchvision.transforms as T
from load_config import load_config, get_parameter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import argparse
import yaml
#import wandb



sys.path.append("..")


image_directory = pathlib.Path("/home/Data/CBIS-DDSM-segmentation-2240x1792/images")
bb = pathlib.Path("/home/Data/CBIS-DDSM-segmentation-2240x1792/bounding_boxes.csv")

 # Esto que es???????????????

parser = argparse.ArgumentParser(description="Train a ddsm patch detector.")
# en el terminal -> python train_mamo.py --config_file base_config.yaml
parser.add_argument("--config_file", type=str, required=True, help="Path to the configuration file.")
parser.add_argument("--overrides", type=str, default = None, help="Overrides for the configuration file.")
#parser.add_argument("--logger", type=str_to_bool, default=True, help="Use wandb for logging.")
args = parser.parse_args()
# Load configuration from YAML file
config = load_config(args.config_file, override_file=args.overrides)

normalize = T.Compose([
    T.ToTensor()
])


# total_dataset = CustomDataset(img_dir = image_directory, labels_file = bb, transform = normalize)

data_module = DDSMPatchDataModule(config=config)
    
    
#logger = get_logger(config) if args.logger else None

# Dividir los elementos en dos conjuntos: 80% para entrenamiento y 20% para prueba
#print("Entrada a la division de los conjuntos")

# Calcular los tamaños de los conjuntos de entrenamiento y prueba
#train_size = int(0.8 * len(total_dataset))
#test_size = len(total_dataset) - train_size

# Dividir el conjunto de datos en entrenamiento y prueba
#train_dataset, test_dataset = random_split(total_dataset, [train_size, test_size])

#train_dataset, test_dataset = train_test_split(total_dataset, test_size=0.2, random_state=42)

#valid_size = int(0.2 * train_size)
#train2_size = train_size - valid_size

seed = torch.Generator().manual_seed(42)
#_, valid_dataset = data.random_split(train_dataset, [train2_size, valid_size], generator = seed)
device = 'cuda'

print("Dataset generados")


#train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn= lambda x: tuple(zip(*x)))
#val_dataloader = DataLoader(valid_dataset, batch_size=2, shuffle=False, collate_fn= lambda x: tuple(zip(*x)))
#test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn= lambda x: tuple(zip(*x)))

print("Dataloader generados")



model = fasterrcnn_resnet50_fpn(num_classes=2, weights_backbone=ResNet50_Weights.IMAGENET1K_V1)

Lmodel = CustomModel(model)


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

trainer.fit(Lmodel, train_dataloader, val_dataloader)


trainer.test(Lmodel, dataloaders = test_dataloader)


print("Training finished")