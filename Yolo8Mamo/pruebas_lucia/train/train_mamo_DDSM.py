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
from pytorch_lightning.loggers import WandbLogger
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.models.detection.anchor_utils import AnchorGenerator




sys.path.append("..")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from load_config import load_config, get_parameter
from classes_dataset_DDSM import DDSM_DataModule
from classes_model_DDSM import DDSM_CustomModel
from classes_dataset_DDSM_3 import DDSM_DataModule_3


def create_callbacks(config):
    callbacks = []
    
    callbacks_dict = get_parameter(config, ["Callbacks"])

    if callbacks_dict is None:
        return callbacks
    
    for callback_name in callbacks_dict:
        if callback_name == "ModelCheckpoint":
            pass
            callbacks.append(ModelCheckpoint(**callbacks_dict[callback_name]))
        elif callback_name == "VisualizeBatchImagesCallback":
            from callbacks import VisualizeBatchImagesCallback
            callbacks.append(VisualizeBatchImagesCallback(**callbacks_dict[callback_name]))

        else:
            raise NotImplementedError(f"Unknown callback {callback_name}")
    return callbacks

# ---------------------------------------------------------------------------------------------



# Training the model
# Para que ejecute en terminal : python train_mamo_DDSM.py --config_file base_config.yaml
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

    # Transformaciones para aumentar el dataset
    



    logger=WandbLogger(project='Mamo')
    
      
    anchor_generator = AnchorGenerator(sizes=((12,),(16,), (32,), (64,), (128,), (256,),(512,), (1024,)), aspect_ratios=((0.5, 1.0, 1.5),)*5)

    model = fasterrcnn_resnet50_fpn(num_classes=2, weights_backbone=ResNet50_Weights.IMAGENET1K_V1, rpn_positive_iou_thresh=0.5,  # umbral para positivas
    rpn_negative_iou_thresh=0.3,  # umbral para negativas
    rpn_anchor_generator = anchor_generator
    )
  
    

# Iterar en el dataset y calclar un histograma de tamaños con ancho, alto y relación de aspecto

    data_module = DDSM_DataModule(config=config) # para 5 clases 
    #data_module = DDSM_DataModule_3(config=config) # para 3 clases
    Lmodel = DDSM_CustomModel(model)






    callbacks = create_callbacks(config)

    trainer_opts = get_parameter(config, ["Trainer"], mode="default", default={})

    for cb in callbacks:
        print(cb, type(cb))

    trainer = L.Trainer( 
        devices='auto',
        accelerator='gpu',
        default_root_dir='/home/lloprib/model_checkpoints',
        callbacks=callbacks, # aquí se pondrá callbacks = callbacks
        logger=logger,
        **trainer_opts
    )

    trainer.fit(Lmodel, data_module)
    #trainer.test(Lmodel, data_module)

    output_model_path = "modelo_entrenado.pth"
    torch.save(Lmodel.state_dict(), output_model_path)
    print(f"Modelo (solo pesos) guardado en: {output_model_path}")


    print("Training finished")
    

