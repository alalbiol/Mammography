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


sys.path.append("..")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from load_config import load_config, get_parameter
from classes_dataset_DDSM import DDSM_DataModule
from classes_model_DDSM import DDSM_CustomModel


# --------------------------------------------------------------------------------------------- corregir

def create_callbacks(config):
    callbacks = []
    
    callbacks_dict = get_parameter(config, ["Callbacks"])
    
    for callback_name in callbacks_dict:
        if callback_name == "EarlyStopping":
            from pytorch_lightning.callbacks import EarlyStopping
            callbacks.append(EarlyStopping(**callbacks_dict[callback_name]))
        elif callback_name == "ModelCheckpoint":
            from pytorch_lightning.callbacks import ModelCheckpoint
            callbacks.append(ModelCheckpoint(**callbacks_dict[callback_name]))
        elif callback_name == "LearningRateMonitor":
            from pytorch_lightning.callbacks import LearningRateMonitor
            callbacks.append(LearningRateMonitor(**callbacks_dict[callback_name]))
        elif callback_name == "LearningRateWarmUp":
            from utils.callbacks import LearningRateWarmUpCallback
            callbacks.append(LearningRateWarmUpCallback(**callbacks_dict[callback_name]))
        elif callback_name == "VisualizeBatchPatches":
            from utils.callbacks import VisualizeBatchPatchesCallback
            callbacks.append(VisualizeBatchPatchesCallback(**callbacks_dict[callback_name]))
        elif callback_name == "GradientNormLogger":
            from utils.callbacks import GradientNormLoggerCallback
            params = callbacks_dict[callback_name] if callbacks_dict[callback_name] is not None else {}
            callbacks.append(GradientNormLoggerCallback(**params))
        elif callback_name == "EMACallback":
            from utils.callbacks import EMACallback
            callbacks.append(EMACallback(**callbacks_dict[callback_name]))
        elif callback_name == "FreezeSwinLayersCallback":
            from utils.callbacks import FreezeSwinLayersCallback
            callbacks.append(FreezeSwinLayersCallback(**callbacks_dict[callback_name]))
        elif callback_name == "LoRACallback":
            from utils.callbacks import LoRACallback
            callbacks.append(LoRACallback(**callbacks_dict[callback_name]))
        elif callback_name == "ReduceLROnEpochsCallback":
            from utils.callbacks import ReduceLROnEpochsCallback
            callbacks.append(ReduceLROnEpochsCallback(**callbacks_dict[callback_name]))

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
    
    train_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Rotate(limit=15, p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=0, p=0.3),
        A.RandomCrop(width=512, height=512, p=0.3),
        A.Resize(640, 640), # reducir las dimensiones a la mitad 
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),  # <-- ¡esto es clave!
        ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))



    logger=WandbLogger(project='Mamo')
    

    model = fasterrcnn_resnet50_fpn(num_classes=2, weights_backbone=ResNet50_Weights.IMAGENET1K_V1)
    data_module = DDSM_DataModule(config=config,transform = train_transforms)
    
    
    #logger = get_logger(config) if args.logger else None


    # Trainer arguments
    #callbacks = create_callbacks(config)
    
    
    # Trainer
    #trainer_kwargs = get_parameter(config, ["Trainer"], mode="default", default={})
    
    #print("Trainer kwargs: ", trainer_kwargs)
    
    Lmodel = DDSM_CustomModel(model)


    checkpoint_callback = ModelCheckpoint( # Para que guarde las tres mejores épocas 
        monitor='train_loss_epoch',
        dirpath='/home/lloprib/',
        filename='mamo-{epoch:02d}-{train_loss_epoch:.2f}',
        save_top_k=3,
        mode='max',
    )

    trainer = L.Trainer(
        max_epochs=50, 
        devices='auto',
        accelerator='gpu',
        default_root_dir='/home/lloprib/proyecto_mam/Mammography/Yolo8Mamo/pruebas_lucia/checkpoints/',
        callbacks=[checkpoint_callback],
        logger=logger,
    )



    trainer.fit(Lmodel, data_module)
    #trainer.test(Lmodel, data_module)


    print("Training finished")
    

