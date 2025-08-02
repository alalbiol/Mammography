import torch
import torchvision
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
import sys
import pandas as pd
import matplotlib.pyplot as plt
import pathlib 
from torch.utils.data import random_split, TensorDataset, DataLoader
import torch.utils.data as data 
import sys
from sklearn.model_selection import train_test_split
import os 
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
from pytorch_lightning.loggers import WandbLogger
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.models.detection.anchor_utils import AnchorGenerator

# Importamos el DataModule y el modelo, y la utilidad para cargar la configuración
sys.path.append("..")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from classes_dataset_DDSM import DDSM_DataModule
from classes_model_DDSM import DDSM_CustomModel 
from load_config import load_config, get_parameter



def run_inference(data_module, output_folder, num_classes, model_path=None):
    """
    Realiza inferencia con un modelo Faster R-CNN en un conjunto de imágenes.
    Además, genera un CSV con las detecciones del modelo.
    """
    print("Cargando el modelo...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_path:
        try:
            anchor_generator = AnchorGenerator(sizes=((12,),(16,), (32,), (64,), (128,), (256,),(512,), (1024,)), aspect_ratios=((0.5, 1.0, 1.5),)*5)
            model = DDSM_CustomModel(fasterrcnn_resnet50_fpn(num_classes=2, weights_backbone=ResNet50_Weights.IMAGENET1K_V1, rpn_positive_iou_thresh=0.5,   # umbral para positivas
            rpn_negative_iou_thresh=0.3,   # umbral para negativas
            rpn_anchor_generator = anchor_generator))
            
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Modelo entrenado cargado desde: {model_path} usando DDSM_CustomModel.")
        except Exception as e:
            print(f"Error al cargar el modelo desde {model_path} usando DDSM_CustomModel: {e}")
            print("Asegúrate de que 'DDSM_CustomModel' se inicializa con los mismos argumentos que durante el entrenamiento.")
            sys.exit(1)
    else:
        model = fasterrcnn_resnet50_fpn(pretrained=True)
        print("Modelo pre-entrenado de COCO cargado.")

    model.eval()
    model.to(device)
    print(f"Modelo listo para inferencia en {device}.")

    data_module.setup(stage="test")
    inference_dataloader = data_module.test_dataloader() 
    if inference_dataloader is None:
        raise ValueError("El DataModule no proporciona un test_dataloader. Asegúrate de que esté implementado.")

    print(f"Iniciando inferencia en el conjunto de test con {len(inference_dataloader)} lotes.")     

    if not os.path.exists(output_folder):
        print(f"Creando carpeta de salida: {output_folder}")
        os.makedirs(output_folder, exist_ok=True)
    else:
        print(f"Carpeta de salida ya existe: {output_folder}")

    all_detections_data = [] 

    image_idx = 0 

    for batch_idx, batch_data in enumerate(inference_dataloader):
        images = batch_data[0] # Esto es una TUPLA de tensores de imagen
        targets = batch_data[1] # Esto es una TUPLA de diccionarios de targets


        images = [img.to(device) for img in images] # Esto convierte la tupla de tensores a lista en el dispositivo.
        
        with torch.no_grad():
            predictions = model(images)


        for i, (image_tensor, prediction) in enumerate(zip(images, predictions)):
            # 'targets' es una tupla, y 'targets[i]' es el diccionario de target para la imagen actual.
            current_image_target = targets[i] 

            # Verificamos la existencia de las claves en el diccionario de target actual.
            if 'original_img_id_str' in current_image_target:
                original_image_id = current_image_target['original_img_id_str']
            elif 'id' in current_image_target: 
                original_image_id = current_image_target['id'].item()
            else:
                original_image_id = f"batch_{batch_idx}_img_{i}"

            image_np = image_tensor.cpu().permute(1, 2, 0).numpy()
            image_pil = Image.fromarray((image_np * 255).astype(np.uint8)) 

            boxes = prediction['boxes'].cpu().numpy()
            labels = prediction['labels'].cpu().numpy()
            scores = prediction['scores'].cpu().numpy()


            for box, score in zip(boxes, scores):
                xmin, ymin, xmax, ymax = box
                all_detections_data.append({
                    'id': original_image_id,
                    'box_x1': xmin,
                    'box_y1': ymin,
                    'box_x2': xmax,
                    'box_y2': ymax,
                    'score': score
                })
            
            if len(boxes) == 0:
                print(f"No se detectaron objetos para la imagen {original_image_id}.")
                image_idx += 1
                continue
            

    print("Inferencia completada en todas las imágenes.")

    detections_df = pd.DataFrame(all_detections_data)
    detections_csv_path = os.path.join(output_folder, "model_detections.csv")
    detections_df.to_csv(detections_csv_path, index=False)
    print(f"\nCSV de detecciones guardado en: {detections_csv_path}")



if __name__ == "__main__":
    output_folder = "/home/lloprib/proyecto_mam/Mammography/Yolo8Mamo/pruebas_lucia/inference/inference_results"

    parser = argparse.ArgumentParser(description="Realiza inferencia con un detector de mamas.")
    parser.add_argument("--config_file", type=str, required=True, help="Ruta al archivo de configuración.")
    args = parser.parse_args()

    config = load_config(args.config_file)
    
    from pprint import pprint
    pprint(config)
    
    GPU_TYPE = get_parameter(config, ["General", "gpu_type"], "None")
    if GPU_TYPE == "RTX 3090":
        torch.set_float32_matmul_precision('medium')
    
    data_module = DDSM_DataModule(config=config)
    data_module.prepare_data()

    mi_modelo_path = "/home/lloprib/proyecto_mam/Mammography/Yolo8Mamo/pruebas_lucia/train/modelo_entrenado.pth"
    
    run_inference(
        data_module=data_module,
        output_folder=output_folder,
        num_classes=5,
        model_path=mi_modelo_path)
    



    