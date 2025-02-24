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

def bounding_boxes_coord(id_imagen, bb):

    """
    Función para iterar sobre las filas de un DataFrame y dibujar rectángulos.
    
    Parameters:
    group_df (DataFrame): DataFrame de un grupo específico.
    """

    bounding_boxes = pd.read_csv(bb)
    nombre_img = f"{id_imagen}.png"  # Concatenar la extensión
    bounding_boxes_grouped = bounding_boxes.groupby("id")
    group_df = bounding_boxes_grouped.get_group(id_imagen)

    cajas=[]

    # Iterar sobre las filas del DataFrame
    for index, row in group_df.iterrows():
        # Obtener los valores de las columnas
        x = row['x']
        y = row['y']
        w = row['w']
        h = row['h']
        
        xmin = x
        ymin = y
        xmax = x + w
        ymax = y + h
        cajas.append([xmin, ymin, xmax, ymax])

    return cajas

class CustomDataset (Dataset): #Voy a crear un dataset personalizado a partir de una caja que ya existe (heredación de clases)
    def __init__(self, img_dir, labels_file, transform=None): #Constructor de la clase, con variables propias de la clase
        self.image_dir = img_dir
        self.labels_file = labels_file
        self.transform = transform



    def __len__(self): #Método que devuelve la longitud del dataset
        bounding_boxes = pd.read_csv(self.labels_file)
        bounding_boxes_grouped = bounding_boxes.groupby("id")
        return len(bounding_boxes_grouped)
    
    def __getitem__(self, idx): #Método que devuelve un item del dataset
        bounding_boxes = pd.read_csv(self.labels_file)
        bounding_boxes_grouped = bounding_boxes.groupby("id")
        img_id = bounding_boxes_grouped.groups.keys()
        img_id = list(img_id)[idx]
        img_name = f"{img_id}.png"
        img_path = os.path.join(self.image_dir, img_name)
        image = np.array(Image.open(img_path).convert("RGB"))
        boxes = bounding_boxes_coord(img_id, self.labels_file)
        boxes = torch.tensor(boxes)
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)
        target = {} # Información de las cajas
        target["boxes"] = boxes
        target["labels"] = labels
        if self.transform:
            image = self.transform(image)
        return image, target

class CustomModel(L.LightningModule): # Creamos un modelo propio a partir de uno que tiene lightning
    # Inicializamos el modelo 
    def __init__(self, model):
        super(CustomModel, self).__init__()
        self.model = model

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
    
    def configure_optimizers(self): # Configuramos el optimizador
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.005, weight_decay = 1e-4, momentum=0.9)
        return optimizer