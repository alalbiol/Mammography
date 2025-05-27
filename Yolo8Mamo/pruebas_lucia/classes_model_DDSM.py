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



def draw_boxes(img, boxes, labels=None, color='red', linewidth=2, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1)
        ax.imshow(img.permute(1, 2, 0).cpu().numpy())

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.tolist()
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                             linewidth=linewidth, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        if labels is not None:
            ax.text(x1, y1, str(labels[i]), color=color, fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.5))
    
    return ax.figure

# _______________________________ESTO ESTÁ BIEN________________________________________________________________________________-

class DDSM_CustomModel(L.LightningModule): # Creamos un modelo propio a partir de uno que tiene lightning
    # Inicializamos el modelo 
    def __init__(self, model):
        super(DDSM_CustomModel, self).__init__()
        self.model = model
        self.map_test=MeanAveragePrecision(iou_type="bbox", extended_summary=True, class_metrics=False, iou_thresholds=[0.5], average='macro')
        self.map=MeanAveragePrecision(iou_type="bbox", extended_summary=True, class_metrics=False, iou_thresholds=[0.5], average='macro')
        self.validation_step_outputs = [] # añadido último



    def forward(self, x):
        # La salida de la red neuronal es la salida del modelo
        return self.model(x)
    
    def training_step(self, batch, batch_idx): # Qué tiene que hacer la red por cada batch
        images, targets = batch

        images = list(image for image in images)  
        
        targets = [{k: v for k, v in t.items()} for t in targets]

        loss_dict = self.model(images, targets) # Como paso imagenes y targets saca la pérdida

        losses = sum(loss for loss in loss_dict.values())

        self.log('train_loss', losses, on_epoch=True, prog_bar=True, batch_size=4, logger=True, on_step=False)
        #on_step = False para que el eje de abscisas de wandb aparezcan las épocas # Sacamos por pantalla la pérdida

        return losses
    
    def validation_step(self, batch, batch_idx): # Qué tiene que hacer la red por cada batch de validación
        images, targets = batch

        images = list(image for image in images)

        targets = [{k: v for k, v in t.items()} for t in targets]

        predictions = self.model(images)
        self.map.update(predictions, targets)



    def on_validation_epoch_end(self):
       
        metrica = self.map.compute()
        self.log("val_map", metrica["map"], on_epoch=True, prog_bar=True, logger=True, batch_size=4)
        self.log("val_precisions", metrica["precision"][0,50,0,0,1], on_epoch=True, prog_bar=True, logger=True, batch_size=4)
        self.log("val_recall", metrica["recall"][0,0,0,1], on_epoch=True, prog_bar=True, logger=True, batch_size=4)

        self.map.reset()

# --------------------------------------------------------------------------

    def on_epoch_end(self):
        if self.current_epoch % 1 == 0:
            self.trainer.validate(self)
    
    def configure_optimizers(self): # Configuramos el optimizador
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, weight_decay = 1e-4, momentum=0.9)
        return optimizer

