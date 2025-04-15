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
from torchvision import transforms, datasets, models
import torchvision.transforms as T


# ___________________ESTO ESTÁ BIEN________________________________________________________________________________________

def bounding_boxes_coord(id_imagen, bb):

    """Función para iterar sobre las filas de un DataFrame y dibujar rectángulos.
    
    Parameters:
    group_df (DataFrame): DataFrame de un grupo específico.
    """

    bounding_boxes = pd.read_csv(bb)
    nombre_img = f"{id_imagen}.png"  # Concatenar la extensión
    bounding_boxes_grouped = bounding_boxes.groupby("id")
    group_df = bounding_boxes_grouped.get_group(id_imagen)

    boxes=[]

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
        boxes.append([xmin, ymin, xmax, ymax])

    return boxes

# ___________________ESTO ESTÁ BIEN________________________________________________________________________________________

class DDSM_CustomDataset (Dataset): #Voy a crear un dataset personalizado a partir de una caja que ya existe (heredación de clases)
    def __init__(self, img_dir, labels_file, transform=None, type=None): #Constructor de la clase, con variables propias de la clase
        self.image_dir = img_dir
        self.labels_file = labels_file
        self.transform = transform
        self.type = type


    def __len__(self): #Método que devuelve la longitud del dataset
        bounding_boxes = pd.read_csv(self.labels_file)
        # bounding_boxes_grouped = bounding_boxes.groupby("id")

        # Extraer la última palabra de cada fila
        bounding_boxes_grouped  = bounding_boxes[bounding_boxes["group"] == self.type]
        bounding_boxes_grouped = bounding_boxes_grouped.groupby("id")



        #bounding_boxes_grouped["last"] = bounding_boxes_grouped["group"].str.split().str[-1]
    
        # Filtrar filas donde la última palabra sea la deseada
        #bounding_boxes_grouped  = bounding_boxes_grouped[bounding_boxes_grouped["last"] == type]

        return len(bounding_boxes_grouped)
    
    def __getitem__(self, idx): #Método que devuelve un item del dataset
        bounding_boxes = pd.read_csv(self.labels_file)
        #bounding_boxes_grouped = bounding_boxes.groupby("id")

        # Extraer la última palabra de cada fila
        bounding_boxes_grouped  = bounding_boxes[bounding_boxes["group"] == self.type]
        bounding_boxes_grouped = bounding_boxes_grouped.groupby("id")

        #bounding_boxes_grouped["last"] = bounding_boxes_grouped["group"].str.split().str[-1]
    
        # Filtrar filas donde la última palabra sea la deseada
        #bounding_boxes_grouped  = bounding_boxes_grouped[bounding_boxes_grouped["last"] == type]

        img_id = bounding_boxes_grouped.groups.keys()
        img_id = list(img_id)[idx]
        img_name = f"{img_id}.png"
        img_path = os.path.join(self.image_dir, img_name)
        image = np.array(Image.open(img_path).convert("RGB"))
        boxes = bounding_boxes_coord(img_id, self.labels_file)
        boxes = torch.tensor(boxes)
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)
        labels = labels.tolist()
        target = {} # Información de las cajas

        if self.transform:
            transformed = self.transform(
                image=image,
                bboxes=boxes,
                category_ids=labels
            )

            image = transformed['image']
            boxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)
            labels = torch.tensor(transformed["category_ids"], dtype=torch.int64)

        target["boxes"] = boxes
        target["labels"] = labels    


        return image, target
    
#_________________________________________________________________________________________________________

#def get_train_dataloader(image_directory, bb, split_csv_train, root_dir,batch_size=32, 
                         #shuffle=True, num_workers=4, return_mask=False):  
def get_train_dataloader(image_directory, bb, batch_size,num_workers, transform=None):
    
    #normalize = T.Compose([ T.ToTensor()])

    #dataset = DDSM_CustomDataset(split_csv, root_dir, return_mask= return_mask, patch_sampler = patch_sampler)
    dataset = DDSM_CustomDataset(img_dir = image_directory, labels_file = bb, transform = transform, type = "train")
    print("AQUI------------------------------------------------")
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn= lambda x: tuple(zip(*x)))
    # collate_fn --> para que funcionen las listas de diccionarios con los batches 
    return dataloader

def get_test_dataloader(image_directory, bb, batch_size,num_workers):

    normalize = T.Compose([ T.ToTensor()]) # PONER EL TO TENSOR DE ALBUMENTATIONS

    dataset = DDSM_CustomDataset(img_dir = image_directory, labels_file = bb, transform = normalize, type = "test")
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn= lambda x: tuple(zip(*x)))
    return dataloader

#_________________________________________________________________________________________________________
class DDSM_DataModule(L.LightningDataModule): # te hace los dataloaders automáticamente 
    def __init__(self, config, transform=None):

        super().__init__()

        #self.labels_file = get_parameter(config, ['Datamodule', 'train_set','labels_file'])
        self.transform = transform

        #self.source_root = get_parameter(config, ['General', 'source_root'], default=None)
        #print(self.source_root)
        
        self.batch_size = get_parameter(config, ['Datamodule',  'batch_size'])
        self.num_workers = get_parameter(config, ['Datamodule', 'num_workers'])
        self.image_directory = get_parameter(config, ['Datamodule','image_directory'])
        self.bb = get_parameter(config, ['Datamodule','bb'])   
       
    
    def train_dataloader(self):
        return get_train_dataloader(image_directory=self.image_directory, 
                                    bb=self.bb, 
                                    batch_size=self.batch_size, 
                                    num_workers=self.num_workers, transform = self.transform) 
        
    
    def test_dataloader(self):
        return get_test_dataloader(image_directory=self.image_directory, 
                                    bb=self.bb, 
                                    batch_size=self.batch_size, 
                                    num_workers=self.num_workers, )