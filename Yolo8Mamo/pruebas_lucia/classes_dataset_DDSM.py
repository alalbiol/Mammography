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
import albumentations as A
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2


# ______________________________________________________________ESTO ESTÁ BIEN_____________________________________________________________

import pandas as pd
import os
from PIL import Image
import numpy as np
import torch

def bounding_boxes_coord(img_id, labels_file):
    """
    Extracts bounding box coordinates for a given image ID from the labels file.
    """
    bounding_boxes = pd.read_csv(labels_file)
    image_boxes = bounding_boxes[bounding_boxes["id"] == img_id]
    boxes = image_boxes[["x_min", "y_min", "x_max", "y_max"]].values.tolist()
    return boxes



# ___________________ESTO ESTÁ BIEN________________________________________________________________________________________

class DDSM_CustomDataset (Dataset): #Voy a crear un dataset personalizado a partir de una caja que ya existe (heredación de clases)
    def __init__(self, img_dir, labels_file, transform=None, type=None): #Constructor de la clase, con variables propias de la clase
        self.image_dir = img_dir
        self.labels_file = labels_file
        self.transform = transform
        self.type = type

        self.bounding_boxes = pd.read_csv(self.labels_file)
        self.grouped_by_id = self.bounding_boxes[self.bounding_boxes["group"] == self.type].groupby("id")
        


        self.img_ids = list(self.grouped_by_id.groups.keys())

        group_sizes = self.grouped_by_id.size()
        # Ahora hacemos un histograma de los tamaños de grupo
        plt.figure(figsize=(8, 5))
        group_sizes.value_counts().sort_index().plot(kind='bar')

        plt.xlabel('Tamaño del grupo')
        plt.ylabel('Número de grupos')
        plt.title('Histograma del tamaño de los grupos por id')
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig('histograma_tamaño_grupos.png') # porque en .py no representa por estar usando el alien



    def bounding_boxes_coord(self, id_imagen):

        """Función para iterar sobre las filas de un DataFrame y dibujar rectángulos.
        
        Parameters:
        group_df (DataFrame): DataFrame de un grupo específico.
        # """

        info_img = self.grouped_by_id.get_group(id_imagen)

        boxes=[]

        # Iterar sobre las filas del DataFrame
        for index, row in info_img.iterrows():
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



    def __len__(self): #Método que devuelve la longitud del dataset
        return len(self.img_ids)

        return len(bounding_boxes_grouped)
    
    def __getitem__(self, idx): #Método que devuelve un item del dataset
        img_id = self.img_ids[idx]
        img_name = f"{img_id}.png"
        img_path = os.path.join(self.image_dir, img_name)
        image = np.array(Image.open(img_path)).astype(np.float32)
        image= convert_16bit_to_8bit(image)  # Convertir a 8 bits
        image = np.expand_dims(image, axis=-1)
        image = np.repeat(image, 3, axis=-1)  # Convertir a 3 canales
        boxes = self.bounding_boxes_coord(img_id)
        boxes = np.array(boxes)
        labels = np.ones((boxes.shape[0],), dtype=np.long)

        target = {} # Información de las cajas
        target["boxes"] = boxes
        target["labels"] = labels

        if self.transform:
            transformed = self.transform(
                image=image,
                bboxes=boxes,
                category_ids=labels.tolist()
            )
            image = transformed['image']
            target["boxes"] = torch.tensor(transformed["bboxes"], dtype=torch.float32)
            target["labels"] = torch.tensor(transformed["category_ids"], dtype=torch.int64)


        return image, target

    
#_________________________________________________________________________________________________________

#def get_train_dataloader(image_directory, bb, split_csv_train, root_dir,batch_size=32, 
                         #shuffle=True, num_workers=4, return_mask=False):  
def get_train_dataloader(image_directory, bb, batch_size,num_workers, transform=None):
    
    #normalize = T.Compose([ T.ToTensor()])

    #dataset = DDSM_CustomDataset(split_csv, root_dir, return_mask= return_mask, patch_sampler = patch_sampler)
    dataset = DDSM_CustomDataset(img_dir = image_directory, labels_file = bb, transform = transform, type = "train")
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn= lambda x: tuple(zip(*x)), shuffle=True) # shuffle --> para que los batches sean aleatorios
    # collate_fn --> para que funcionen las listas de diccionarios con los batches 
    return dataloader

def get_test_dataloader(image_directory, bb, batch_size,num_workers, transform=None):
    dataset = DDSM_CustomDataset(img_dir = image_directory, labels_file = bb, transform = transform, type = "test")
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn= lambda x: tuple(zip(*x)), shuffle=True)
    return dataloader


def convert_16bit_to_8bit(image_16bit):
    """Convierte una imagen de 16 bits (NumPy array) a 8 bits."""


    min_16bit = np.min(image_16bit)
    max_16bit = np.max(image_16bit)


    if max_16bit > min_16bit:
        # Escalar al rango 0-1
        image_scaled = (image_16bit - min_16bit) / (max_16bit - min_16bit)

        # Escalar al rango 0-255 y convertir a uint8
        image_scaled = (image_scaled ).astype(np.float32)
    else:
        # Si todos los valores son iguales, crear una imagen de 8 bits con ese valor (o 0)
        image_scaled = np.full_like(image_16bit, 0, dtype=np.float32)

    return image_scaled

#_________________________________________________________________________________________________________
class DDSM_DataModule(L.LightningDataModule): # te hace los dataloaders automáticamente 
    def __init__(self, config):

        super().__init__()
        
        self.batch_size = get_parameter(config, ['Datamodule',  'batch_size'])
        self.num_workers = get_parameter(config, ['Datamodule', 'num_workers'])
        self.image_directory = get_parameter(config, ['Datamodule','image_directory'])
        self.bb = get_parameter(config, ['Datamodule','bb'])   

        self.train_transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Rotate(limit=35, p=0.3),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=0, p=0.3),
            A.RandomSizedBBoxSafeCrop(width=512, height=512, p=0.3, erosion_rate=0.2),
            #A.Resize(2240, 1792), # reducir las dimensiones a la mitad 
            #A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),  # <-- ¡esto es clave!
            ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids'], min_visibility = 0.3)) # min_visibility es el porcentaje mínimo de visibilidad de la caja para que se considere válida

        self.val_transforms = A.Compose([
            A.Resize(2240, 1792), # reducir las dimensiones a la mitad 
            #A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),  # <-- ¡esto es clave!
            ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))
    
    def train_dataloader(self):
        return get_train_dataloader(image_directory=self.image_directory, 
                                    bb=self.bb, 
                                    batch_size=self.batch_size, 
                                    num_workers=self.num_workers, transform = self.train_transforms) 
    def val_dataloader(self):
        return get_test_dataloader(image_directory=self.image_directory, 
                                    bb=self.bb, 
                                    batch_size=self.batch_size, 
                                    num_workers=self.num_workers, transform = self.val_transforms) 
    
    def test_dataloader(self):
        return get_test_dataloader(image_directory=self.image_directory, 
                                    bb=self.bb, 
                                    batch_size=self.batch_size, 
                                    num_workers=self.num_workers, transform = self.val_transforms)