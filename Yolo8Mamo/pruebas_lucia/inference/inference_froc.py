import torch
import torchvision # para cargar modelos preentrenados
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image # para cargar y guardar imágenes
import os # para operaciones de sistema de archivos
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import albumentations as A # para transformaciones de imágenes
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
import sys
import argparse # para manejar argumentos de línea de comandos


# cosas que cambiar: las filtraciones por el umbral de confianza hay que cambiarlas para hacer lo que queremos

# Importamos el DataModule
sys.path.append("..")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from classes_dataset_DDSM import DDSM_DataModule
from classes_model_DDSM import DDSM_CustomModel
from load_config import load_config, get_parameter # para cargar la configuración del modelo


def run_inference(data_module, output_folder, model_path=None, threshold=0.5):

    """
    Realiza inferencia con un modelo Faster R-CNN en un conjunto de imágenes.

    Args:
        data_module (pytorch_lightning.core.datamodule.LightningDataModule): Instancia del DataModule ya configurado 
        output_folder (str): Ruta a la carpeta donde se guardarán las imágenes con detecciones.
        model_path (str, optional): Ruta al archivo .pth del modelo entrenado. 
                                    Si es None, se carga un modelo pre-entrenado de TorchVision.
        threshold (float, optional): Umbral de confianza para mostrar las detecciones. 
                                     Las detecciones con una puntuación inferior a este umbral se filtrarán.
    """

    # 1. CARGAR EL MODELO
    print("Cargando el modelo...")
    if model_path:
        # Cargar un modelo entrenado por ti
        model = fasterrcnn_resnet50_fpn(pretrained=False) # se carga un modelo que no ha sido preentrenado
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_classes = 5 # Asumimos 4 clases (calcificación maligna, calcificación benigna, masa maligna, masa benigna) + 1 para el fondo
        # ¿Esto está bien???????????????

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # Ahora el modelo está listo para un entrenamiento de fine-tuning o inferencia
        model.eval() # O model.train() si vas a entrenar
        model.to(device)

        print("Modelo pre-entrenado de COCO cargado y cabeza de clasificador ajustada.")


    else:
        # Cargar un modelo pre-entrenado de TorchVision (entrenado en COCO)
        model = fasterrcnn_resnet50_fpn(pretrained=True)


    
    model.eval() # Poner el modelo en modo evaluación
    # Configura el dispositivo (GPU si está disponible, de lo contrario CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    model.to(device)
    print(f"Modelo cargado y movido a {device}.")


    # 2. OBTENER EL DATALOADER DE VALIDACIÓN DEL DATAMODULE
    val_dataloader = data_module.val_dataloader()
    if val_dataloader is None:
        raise ValueError("El DataModule no proporciona un val_dataloader. Asegúrate de que esté implementado.")

    print(f"Iniciando inferencia en el conjunto de validación con {len(val_dataloader)} lotes.")    

    # DEFINIR LAS TRANSFORMACIONES PARA LAS IMÁGENES
   # transform = A.Compose([
         #   A.Resize(2240, 1792), ToTensorV2()], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))
    

    # 3. PREPARAR EL DIRECTORIO DE SALIDA
    if not os.path.exists(output_folder): # crea la carpeta donde se guardan las imágenes con las detecciones si ésta no existe
        print(f"Creando carpeta de salida: {output_folder}")
        os.makedirs(output_folder, exist_ok=True)
    else:
        print(f"Carpeta de salida ya existe: {output_folder}")

    # 4. ITERAR SOBRE EL DATALOADER DE VALIDACIÓN

    image_idx = 0 # Contador para nombrar las imágenes de salida

    for batch_idx, (images, targets) in enumerate(val_dataloader):
        # 'images' es una lista de tensores (una imagen por entrada)
        # 'targets' es una lista de diccionarios, con las anotaciones
        
        # Mover las imágenes al dispositivo
        images = [img.to(device) for img in images]
        # Realizar la inferencia
        with torch.no_grad():
            predictions = model(images) # predictions es una lista de diccionarios, uno por imagen en el lote


        # 5. PROCESAR CADA IMAGEN EN EL LOTE
        for i, (image_tensor, prediction) in enumerate(zip(images, predictions)):

            image_np = image_tensor.cpu().permute(1, 2, 0).numpy()
            image_pil = Image.fromarray((image_np * 255).astype(np.uint8)) # Asume que está en [0,1]

            boxes = prediction['boxes'].cpu().numpy()
            labels = prediction['labels'].cpu().numpy()
            scores = prediction['scores'].cpu().numpy()

            # Filtrar detecciones por umbral de confianza
            keep = scores > threshold
            boxes = boxes[keep]
            labels = labels[keep]
            scores = scores[keep]


        
        # 7. VISUALIZAR LAS DETECCIONES

        if len(boxes) == 0:
            print(f"No se detectaron objetos con confianza > {threshold}.")
            continue
        fig, ax = plt.subplots(1)
        ax.imshow(image_np)

        for box, label, score in zip(boxes, labels, scores):
            # Las cajas están en formato [x1, y1, x2, y2]
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1

            # Crear un rectángulo y añadirlo a la imagen
            rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

            # Añadir etiqueta y puntuación. Solo se muestra el ID numérico y la puntuación.
            plt.text(x1, y1 - 10, f'ID:{label} - Conf:{score:.2f}', color='red', fontsize=8, 
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0))

        ax.axis('off')

        output_image_name = f"det_image_{image_idx:04d}.jpg" 
        full_output_path = os.path.join(output_folder, output_image_name)

        plt.savefig(full_output_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig) # se cierra la figura para ahorrar memoria
        print(f"Resultados guardados en: {full_output_path}")
        image_idx += 1 # se incrementa el contador para la siguiente imagen


    print("Inferencia completada en todas las imágenes.")

if __name__ == "__main__":

    output_folder = "/home/lloprib/proyecto_mam/Mammography/Yolo8Mamo/pruebas_lucia/inference/inference_results" # Carpeta donde se guardan las imágenes con detecciones

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
   
    

    data_module = DDSM_DataModule(config=config)
    
    data_module.prepare_data()
    data_module.setup(stage="test")  

    

    # Si quieres usar tu propio modelo entrenado, descomenta y ajusta la siguiente línea:
    mi_modelo = "/home/lloprib/proyecto_mam/Mammography/Yolo8Mamo/pruebas_lucia/train/modelo_entrenado.pth"
    #my_trained_model_path = None # Si se quiere usar el modelo pre-entrenado de TorchVision
    

    run_inference(
        data_module=data_module,
        output_folder=output_folder,
        model_path=mi_modelo,
        threshold=0.3,
    )