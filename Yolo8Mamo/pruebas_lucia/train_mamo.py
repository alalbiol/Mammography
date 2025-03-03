#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
import matplotlib.patches as patches
from clases_train_mamo import bounding_boxes_coord, CustomDataset, CustomModel
from torch.utils.data import DataLoader 
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models import ResNet50_Weights
import lightning as L
import torchvision.transforms as T


sys.path.append("..")


# In[ ]:


image_directory = pathlib.Path("/home/Data/CBIS-DDSM-segmentation-2240x1792/images")
bb = pathlib.Path("/home/Data/CBIS-DDSM-segmentation-2240x1792/bounding_boxes.csv")

normalize = T.Compose([
    T.ToTensor()
])


total_dataset = CustomDataset(img_dir = image_directory, labels_file = bb, transform = normalize)

# Dividir los elementos en dos conjuntos: 80% para entrenamiento y 20% para prueba
print("Entrada a la division de los conjuntos")

# Calcular los tamaños de los conjuntos de entrenamiento y prueba
train_size = int(0.8 * len(total_dataset))
test_size = len(total_dataset) - train_size

# Dividir el conjunto de datos en entrenamiento y prueba
train_dataset, test_dataset = random_split(total_dataset, [train_size, test_size])

#train_dataset, test_dataset = train_test_split(total_dataset, test_size=0.2, random_state=42)

valid_size = int(0.2 * train_size)
train2_size = train_size - valid_size

seed = torch.Generator().manual_seed(42)
_, valid_dataset = data.random_split(train_dataset, [train2_size, valid_size], generator = seed)
device = 'cuda'

print("Dataset generados")


# In[ ]:


print(len(train_dataset))


# In[ ]:


train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn= lambda x: tuple(zip(*x)))
valid_dataloader = DataLoader(valid_dataset, batch_size=2, shuffle=False, collate_fn= lambda x: tuple(zip(*x)))
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn= lambda x: tuple(zip(*x)))

print("Dataloader generados")


# In[ ]:


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
    max_epochs=100, 
    devices='auto',
    accelerator='gpu',
    default_root_dir='/home/lloprib/proyecto_mam/Mammography/Yolo8Mamo/pruebas_lucia/checkpoints/',
    callbacks=[checkpoint_callback]
)

trainer.fit(Lmodel, train_dataloader, valid_dataloader)


trainer.test(Lmodel, dataloaders = test_dataloader)


print("Training finished")

