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
from torch.utils.data import Dataset, DataLoader
from torchmetrics.detection import MeanAveragePrecision
from torchvision import transforms, datasets, models
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2



sys.path.append("..")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from proy_vgg16.configs.load_config import load_config, get_parameter


class CBIS_DDSM_CustomDataset (Dataset): #Voy a crear un dataset personalizado a partir de una caja que ya existe (heredación de clases)

    def bounding_boxes_coord(self, img_id):
        bounding_boxes = self.bounding_boxes
        image_boxes = bounding_boxes[bounding_boxes["id"] == img_id]
        boxes = image_boxes[["minx", "miny", "maxx", "maxy"]].values.tolist()
        return boxes
    

    def load_image(self,path, dataset='CBIS-DDSM'):
        if dataset == 'CBIS-DDSM':
            image = np.array(Image.open(path)).astype(np.float32)
            im = image / image.max() * 3.0
            image = np.clip(im, 0.8, 2.9) - 0.8
            image = (255. * image / image.max())
            image = np.clip(image, 0, 255)
        return image
    
    def resize_image(self, image, target_scale, max_size):
        im_shape = image.shape
        im_size_min = np.min(im_shape[:2])
        im_size_max = np.max(im_shape[:2])

        scale = float(self.target_scale) / float(im_size_min)
        # Prevent the longer side from exceeding max_size
        if np.round(scale * im_size_max) > self.max_size:
            scale = float(self.max_size) / float(im_size_max)

        # Resize image with computed scale
        image_resized = cv2.resize(image, (0, 0), fx=scale, fy=scale)

        return image_resized, scale 


    def __init__(self, img_dir, labels_file, transform=None, type=None, target_scale=1700, max_size=2100): #Constructor de la clase, con variables propias de la clase
        self.image_dir = img_dir
        self.labels_file = labels_file
        self.transform = transform
        self.type = type
        self.target_scale = target_scale
        self.max_size = max_size

        self.bounding_boxes = pd.read_csv(self.labels_file)
        self.grouped_by_id = self.bounding_boxes[self.bounding_boxes["group"] == self.type].groupby("id")

        self.transform = A.Compose([
            A.SmallestMaxSize(max_size = target_scale),
            A.LongestMaxSize(max_size = max_size),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        

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


    def __len__(self): #Método que devuelve la longitud del dataset
        return len(self.img_ids)

    
    def __getitem__(self, idx): #Método que devuelve un item del dataset
        img_id = self.img_ids[idx]
        img_name = f"{img_id}.png"
        img_path = os.path.join(self.image_dir, img_name)


        image = self.load_image(img_path, dataset='CBIS-DDSM') 

        #image= convert_16bit_to_8bit(image)  # Convertir a 8 bits ¿esto lo tengo que seguir haciendo?
        #image = np.expand_dims(image, axis=-1)
        #image = np.repeat(image, 3, axis=-1)  # Convertir a 3 canales

        #image_resized, scale = self.resize_image(image, target_scale=self.target_scale, max_size=self.max_size)
        


        # funcion --> normalize intensity

        image = np.stack((image,) * 3, axis=-1)  # Convert grayscale to RGB by repeating channels
        pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]]).reshape((1, 1, 3))  # Mean pixel values for VGG16
        image = image - pixel_means  # Subtract mean pixel
        # image_resized = torch.from_numpy(image_resized).permute(2, 0, 1).float() # (H, W, C) -> (C, H, W) y a float

        # _________________________________________________________


        #_____________________________________________________________



        image_boxes = self.bounding_boxes[self.bounding_boxes["id"] == img_id]

        boxes = self.bounding_boxes_coord(img_id)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)  # Convertir a tensor de tipo float32

        #-------------------------------------------------

        # h, w = image.shape[:2]
        # new_h, new_w = image_resized.shape[:2]
        # scale_x = new_w / w
        # scale_y = new_h / h
        # boxes[:, [0, 2]] *= scale_x
        # boxes[:, [1, 3]] *= scale_y

        #-------------------------------------------------


        #boxes = boxes * scale  # Escalar los bounding boxes
        labels = image_boxes["cancer"].values
        labels = torch.as_tensor(labels, dtype=torch.int64)+1
        labels = labels.tolist()

        if self.transform:
            transformed = self.transform(image=image, 
                                         bboxes=boxes,
                                         labels=labels
                                         ) # esto es una lista
            
        image_resized = transformed['image'] # esto extrae los elementos de la lista
        image_resized = torch.as_tensor(image_resized, dtype=torch.float32)
        boxes = torch.tensor(transformed['bboxes'], dtype=torch.float32)
        target = {
            "boxes": boxes,                 
            "labels": torch.tensor(labels, dtype=torch.int64),           
            "image_id": torch.tensor([idx], dtype=torch.int64)
        }

        target["id"] = torch.tensor([idx], dtype=torch.int64)  # Añadir el índice de la imagen como ID

       
        #---------------------------------------------------


        #---------------------------------------------------


        return image_resized, target

    
#_________________________________________________________________________________________________________
 
def get_train_dataloader(image_directory, bb, batch_size,num_workers, transform=None, target_scale=1700, max_size=2100):
    
    #normalize = T.Compose([ T.ToTensor()])

    dataset = CBIS_DDSM_CustomDataset(img_dir = image_directory, labels_file = bb, transform = None, type = "train", target_scale=target_scale, max_size=max_size)
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn= lambda x: tuple(zip(*x)), shuffle=True) # shuffle --> para que los batches sean aleatorios
    # collate_fn --> para que funcionen las listas de diccionarios con los batches 
    return dataloader

def get_test_dataloader(image_directory, bb, batch_size,num_workers, transform=None):
    dataset = CBIS_DDSM_CustomDataset(img_dir = image_directory, labels_file = bb, transform = None, type = "test")
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn= lambda x: tuple(zip(*x)), shuffle=True)
    return dataloader


# def convert_16bit_to_8bit(image_16bit):
#     """Convierte una imagen de 16 bits (NumPy array) a 8 bits."""

#     min_16bit = np.min(image_16bit)
#     max_16bit = np.max(image_16bit)


#     if max_16bit > min_16bit:
#         # Escalar al rango 0-1
#         image_scaled = (image_16bit - min_16bit) / (max_16bit - min_16bit)

#         # Escalar al rango 0-255 y convertir a uint8
#         image_scaled = (image_scaled ).astype(np.float32)
#     else:
#         # Si todos los valores son iguales, crear una imagen de 8 bits con ese valor (o 0)
#         image_scaled = np.full_like(image_16bit, 0, dtype=np.float32)

#     return image_scaled

#_________________________________________________________________________________________________________

class CBIS_DDSM_DataModule(L.LightningDataModule): 
    def __init__(self, config):

        super().__init__()
        
        self.batch_size = get_parameter(config, ['Datamodule',  'batch_size'])
        self.num_workers = get_parameter(config, ['Datamodule', 'num_workers'])
        self.image_directory = get_parameter(config, ['Datamodule','image_directory'])
        self.bb = get_parameter(config, ['Datamodule','bb'])   

        # Parámetros del estilo Faster-RCNN/VGG16
        self.scales = get_parameter(config, ['Trainer', 'SCALES'])[0]  
        self.max_size = get_parameter(config, ['Trainer', 'MAX_SIZE'])

    
    def train_dataloader(self):
        return get_train_dataloader(image_directory=self.image_directory, 
                                    bb=self.bb, 
                                    batch_size=self.batch_size, 
                                    num_workers=self.num_workers,
                                    target_scale=self.scales, 
                                    max_size=self.max_size,
                                    transform = None
        ) 
    def val_dataloader(self):
        return get_test_dataloader(image_directory=self.image_directory, 
                                    bb=self.bb, 
                                    batch_size=self.batch_size, 
                                    num_workers=self.num_workers, transform = None) 
    
    def test_dataloader(self):
        return get_test_dataloader(image_directory=self.image_directory, 
                                    bb=self.bb, 
                                    batch_size=self.batch_size, 
                                    num_workers=self.num_workers, transform = None)