from torch.utils.data import Dataset 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pathlib 
import sys
import os 
import matplotlib.patches as patches
import torch
import lightning as L
from torchmetrics.detection import MeanAveragePrecision
from load_config import load_config, get_parameter
from torch.utils.data import Dataset, DataLoader




# _______________________________ESTO ESTÁ BIEN________________________________________________________________________________-

class DDSM_CustomModel(L.LightningModule): # Creamos un modelo propio a partir de uno que tiene lightning
    # Inicializamos el modelo 
    def __init__(self, model):
        super(DDSM_CustomModel, self).__init__()
        self.model = model
        self.map_test=MeanAveragePrecision(iou_type="bbox", extended_summary=True, class_metrics=True)
        self.map=MeanAveragePrecision(iou_type="bbox", extended_summary=True, class_metrics=True)

    def forward(self, x):
        # La salida de la red neuronal es la salida del modelo
        return self.model(x)
    
    def training_step(self, batch, batch_idx): # Qué tiene que hacer la red por cada batch
        images, targets = batch

        images = list(image for image in images)
        targets = [{k: v for k, v in t.items()} for t in targets]

        loss_dict = self.model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        self.log('train_loss', losses, on_epoch=True, prog_bar=True) # Sacamos por pantalla la pérdida

        return losses
    
    def validation_step(self, batch, batch_idx): # Qué tiene que hacer la red por cada batch de validación
        images, targets = batch

        images = list(image for image in images)
        targets = [{k: v for k, v in t.items()} for t in targets]

        predictions = self.model(images)
        self.map.update(predictions, targets)

    def on_validation_epoch_end(self):
        metrica = self.map.compute()

        self.log("val_map", metrica["map"], on_epoch=True, prog_bar=True)
        self.log("val_precisions", metrica["map_75"], on_epoch=True, prog_bar=True)
        self.log("val_recall", metrica["mar_100_per_class"], on_epoch=True, prog_bar=True)

        self.map.reset()

    def test_step(self, batch, batch_idx): # Qué tiene que hacer la red por cada batch de test
        images, targets = batch
        images=list(image for image in images)
        targets = [{k: v for k, v in t.items()} for t in targets]
        predictions = self.model(images)
        self.map_test.update(predictions, targets)

    def on_test_epoch_end(self):
        metrica = self.map_test.compute()
        self.log("test_map", metrica["map"], on_epoch=True, prog_bar=True)
        self.log("test_precisions", metrica["map_75"], on_epoch=True, prog_bar=True)
        self.log("test_recall", metrica["mar_100_per_class"], on_epoch=True, prog_bar=True)
        self.map_test.reset()

    def on_epoch_end(self):
        if self.current_epoch % 1 == 0:
            self.trainer.validate(self)
    
    def configure_optimizers(self): # Configuramos el optimizador
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.005, weight_decay = 1e-4, momentum=0.9)
        return optimizer