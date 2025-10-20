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
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator


sys.path.append("..")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from proy_vgg16.configs.load_config import load_config, get_parameter
from proy_vgg16.models.backbone_vgg16 import VGG16Backbone


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


DEBUG = False

def load_caffe_weights_into_pytorch(pytorch_model, extracted_weights_dir):
    """
    Esta función intenta cargar pesos extraçidos de Caffe (guardados como archivos .npy, uno por peso y otro por bias por capa) y copiarlos dentro del state_dict() de un modelo PyTorch. 
    Para ello, usa un mapa que relaciona nombres de capas en Caffe con prefijos/paths de parámetros en el modelo Pytorch. 
    
    """
    print(f"\nCargando pesos de Caffe desde '{extracted_weights_dir}' al modelo PyTorch...")

    # Mapping Caffe layer names to PyTorch module names and parameter names
    # This is crucial for correctly mapping the weights.
    # Caffe: {layer_name}_weights.npy, {layer_name}_biases.npy
    # PyTorch: module_name.weight, module_name.bias

    # VGG Backbone mapping

    # Los archivos .npy se buscarán con nombres como conv1_1_weights.npy, conv1_1_biases.npy, etc. y luego se intentarán copiar a backbone.conv1_1.weight y backbone.conv1_1.bias

    caffe_to_pytorch_map = {
        'conv1_1': 'backbone.conv1_1', 'conv1_2': 'backbone.conv1_2',
        'conv2_1': 'backbone.conv2_1', 'conv2_2': 'backbone.conv2_2',
        'conv3_1': 'backbone.conv3_1', 'conv3_2': 'backbone.conv3_2', 'conv3_3': 'backbone.conv3_3',
        'conv4_1': 'backbone.conv4_1', 'conv4_2': 'backbone.conv4_2', 'conv4_3': 'backbone.conv4_3',
        'conv5_1': 'backbone.conv5_1', 'conv5_2': 'backbone.conv5_2', 'conv5_3': 'backbone.conv5_3',
        'rpn_conv/3x3': 'rpn.rpn_conv',
        'rpn_cls_score': 'rpn.rpn_cls_score',
        'rpn_bbox_pred': 'rpn.rpn_bbox_pred',
        'fc6': 'rcnn_head.fc6',
        'fc7': 'rcnn_head.fc7',
        'cls_score': 'rcnn_head.cls_score',
        'bbox_pred': 'rcnn_head.bbox_pred',
    }

    model_state_dict = pytorch_model.state_dict() # diccionario con todos los parámetros esperados por el modelo
    print("Número de parámetros en el modelo de Pytorch:", len(model_state_dict))

    num_imported_weights = 0 # para contar cuántos parámetros se cargan
    for caffe_layer_name, pytorch_module_prefix in caffe_to_pytorch_map.items():
        # Carga pesos .npy
        weights_path = os.path.join(extracted_weights_dir, f"{caffe_layer_name}_weights.npy")
        if os.path.exists(weights_path):
            caffe_weights = np.load(weights_path)
            # Caffe convolutional weights are (out_channels, in_channels, kH, kW)
            # PyTorch convolutional weights are (out_channels, in_channels, kH, kW) - direct match!
            # Caffe InnerProduct weights are (out_channels, in_channels)
            # PyTorch Linear weights are (out_features, in_features) - direct match!
            # Ensure the shapes match before loading
            try:
                # Detecta si hay entrada .weight en el state_dict del modelo PyTorch
                if f"{pytorch_module_prefix}.weight" in model_state_dict:
                    # Si los pesos son convolucionales (4D)
                    if len(caffe_weights.shape) == 4:
                        # Caffe weights (out, in, h, w)
                        # PyTorch weights (out, in, h, w)
                        # Compara la forma esperada y si coincide copia directamente, si no copia pero salta un aviso
                        expected_shape = model_state_dict[f"{pytorch_module_prefix}.weight"].shape
                        if caffe_weights.shape != expected_shape:
                            print(f"Warning: Weight shape mismatch for {caffe_layer_name}: Caffe {caffe_weights.shape} vs PyTorch {expected_shape}. Transposing if necessary.")
                            # Attempt to transpose if it's a common Caffe-to-PyTorch conv weight mismatch (unlikely for VGG)
                            # E.g., Caffe (out, in, kH, kW) -> PyTorch (out, in, kH, kW)
                            # If they don't match, you might need to investigate specific layer configurations.
                            # For VGG, it should generally be a direct match.
                        model_state_dict[f"{pytorch_module_prefix}.weight"].copy_(torch.from_numpy(caffe_weights))
                        num_imported_weights += 1
                        if DEBUG:
                            print(f"  Loaded {caffe_layer_name}_weights into {pytorch_module_prefix}.weight")

                    # For InnerProduct (Linear) layers (si son pesos de fully-connected (2D))
                    elif len(caffe_weights.shape) == 2:
                        # Caffe FC weights are (out_features, in_features)
                        # PyTorch Linear weights are (out_features, in_features)
                        # So, a direct copy is usually sufficient.
                        # However, for VGG, the fully connected layers typically
                        # connect from a flattened convolutional output.
                        # Caffe's InnerProduct can behave like a matrix multiplication
                        # where the input is flattened.
                        # PyTorch's Linear layer expects a 2D input (batch_size, in_features).
                        # The Caffe weights are often stored as (output_dim, input_dim).
                        # PyTorch's Linear.weight is (out_features, in_features).
                        # For VGG's fc6/fc7, Caffe's shape is usually (output_channels, input_channels).
                        # PyTorch's Linear expects (output_features, input_features).
                        # Often, Caffe stores FC weights as (out_dim, in_dim).
                        # PyTorch's `nn.Linear` `weight` is (out_features, in_features).
                        # So, we need to transpose Caffe's (in_dim, out_dim) to (out_dim, in_dim) if that's the case.
                        # Let's check the common VGG FC behavior.
                        # For VGG, Caffe's FC weights are usually (output_dim, input_dim).
                        # PyTorch's `nn.Linear` weights are also (output_dim, input_dim).
                        # So, direct copy for FC layers should be fine.
                        expected_shape = model_state_dict[f"{pytorch_module_prefix}.weight"].shape
                        if caffe_weights.shape != expected_shape:
                             # This is a common point of failure for Caffe FC to PyTorch Linear
                             # Caffe's InnerProduct might have its weights stored in a different order (e.g., (in_dim, out_dim))
                             # or might expect a flattened input in a different channel order.
                             # If a mismatch occurs, try transposing.
                             # Cmpara formas y si coinciden copia. Si no coinciden, el código intenta avisar y transponer (por el sys se cierra directamente antes de transponer)
                             print(f"Warning: FC weight shape mismatch for {caffe_layer_name}: Caffe {caffe_weights.shape} vs PyTorch {expected_shape}. Attempting transpose.")
                             import sys
                             sys.exit()
                             try:
                                 model_state_dict[f"{pytorch_module_prefix}.weight"].copy_(torch.from_numpy(caffe_weights.T))
                                 print(f"  Loaded {caffe_layer_name}_weights into {pytorch_module_prefix}.weight (transposed)")
                                 num_imported_weights += 1
                             except Exception as transpose_e:
                                 print(f"  Failed to transpose and load: {transpose_e}")
                                 print(f"  Please manually verify weight dimensions for {caffe_layer_name}.")
                        else:
                            model_state_dict[f"{pytorch_module_prefix}.weight"].copy_(torch.from_numpy(caffe_weights))
                            if DEBUG:
                                print(f"  Loaded {caffe_layer_name}_weights into {pytorch_module_prefix}.weight")
                            num_imported_weights += 1

            # Manejo de errores al copiar los pesos
            except RuntimeError as e:
                print(f"Error loading weights for {caffe_layer_name} (into {pytorch_module_prefix}.weight): {e}")
                print(f"Caffe weights shape: {caffe_weights.shape}, PyTorch expected shape: {model_state_dict[f'{pytorch_module_prefix}.weight'].shape}")

        # Carga biases
        biases_path = os.path.join(extracted_weights_dir, f"{caffe_layer_name}_biases.npy")
        if os.path.exists(biases_path):
            caffe_biases = np.load(biases_path)
            try:
                model_state_dict[f"{pytorch_module_prefix}.bias"].copy_(torch.from_numpy(caffe_biases))
                if DEBUG:
                    print(f"  Loaded {caffe_layer_name}_biases into {pytorch_module_prefix}.bias")
                num_imported_weights += 1
            except RuntimeError as e:
                print(f"Error loading biases for {caffe_layer_name} (into {pytorch_module_prefix}.bias): {e}")
                print(f"Caffe biases shape: {caffe_biases.shape}, PyTorch expected shape: {model_state_dict[f'{pytorch_module_prefix}.bias'].shape}")

    # Actualización del modelo (reaplica el state_dict modificado al modelo y luego imprime un resumen)
    pytorch_model.load_state_dict(model_state_dict)
    print("All available Caffe weights loaded into the PyTorch model.")
    print(f"Total parameters imported: {num_imported_weights} out of {len(model_state_dict)}")


class Faster_VGG16_CustomModel(L.LightningModule): 
    def __init__(self, num_classes, config):

        super().__init__()
        self.config = config
        self.backbone = VGG16Backbone()

        
        # Crear el AnchorGenerator.
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )
    

        # Usa el parámetro del config para la ruta de los pesos
        self.EXTRACTED_WEIGHTS_DIR = getattr(config["Model"]["weights_directory"], "WEIGHTS_DIR", "")

        self.pytorch_model = FasterRCNN(
            backbone=self.backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            min_size=config["Trainer"]["SCALES"][0],
            max_size=config["Trainer"]["MAX_SIZE"],
            rpn_pre_nms_top_n_train=config["Trainer"]["RPN_PRE_NMS_TOP_N"],
            rpn_pre_nms_top_n_test=config["Test"]["RPN_POST_NMS_TOP_N"],
            rpn_post_nms_top_n_train=config["Trainer"]["RPN_POST_NMS_TOP_N"],
            rpn_post_nms_top_n_test=config["Test"]["RPN_POST_NMS_TOP_N"],
            rpn_nms_thresh=config["Trainer"]["RPN_NMS_THRESH"],
            rpn_fg_iou_thresh=config["Trainer"]["RPN_POSITIVE_OVERLAP"],
            rpn_bg_iou_thresh=config["Trainer"]["RPN_NEGATIVE_OVERLAP"],
            rpn_batch_size_per_image=config["Trainer"]["RPN_BATCHSIZE"],
            rpn_positive_fraction=config["Trainer"]["RPN_FG_FRACTION"],
            box_score_thresh=config["Trainer"]["BBOX_THRESH"],
            box_nms_thresh=config["Trainer"]["RPN_NMS_THRESH"],
            box_fg_iou_thresh=config["Trainer"]["RPN_POSITIVE_OVERLAP"],
            box_bg_iou_thresh=config["Trainer"]["RPN_NEGATIVE_OVERLAP"],
            box_batch_size_per_image=config["Trainer"]["RPN_BATCHSIZE"]
        )
        
        # Solo inicializa los pesos si la ruta no está vacía
        if self.EXTRACTED_WEIGHTS_DIR:
            load_caffe_weights_into_pytorch(self.pytorch_model, self.EXTRACTED_WEIGHTS_DIR)
        
        self.map_train = MeanAveragePrecision(iou_type="bbox", extended_summary=True, class_metrics=False, iou_thresholds=[0.5], average='macro')
        self.map_test = MeanAveragePrecision(iou_type="bbox", extended_summary=True, class_metrics=False, iou_thresholds=[0.5], average='macro')
        self.map = MeanAveragePrecision(iou_type="bbox", extended_summary=True, class_metrics=False, iou_thresholds=[0.5], average='macro')
        self.validation_step_outputs = []

    def forward(self, x):
        # La salida de la red neuronal es la salida del modelo
        return self.pytorch_model(x)
    
    def training_step(self, batch, batch_idx): # Qué tiene que hacer la red por cada batch
        images, targets = batch

        images = list(image for image in images)  
        
        targets = [{k: v for k, v in t.items()} for t in targets]

        loss_dict = self.pytorch_model(images, targets) # Como paso imagenes y targets saca la pérdida

        losses = sum(loss for loss in loss_dict.values())

        self.log('train_loss', losses, on_epoch=True, prog_bar=True, batch_size=128, logger=True, on_step=False)
        #on_step = False para que el eje de abscisas de wandb aparezcan las épocas # Sacamos por pantalla la pérdida

        return losses
    
    def validation_step(self, batch, batch_idx): # Qué tiene que hacer la red por cada batch de validación
        images, targets = batch

        images = list(image for image in images)

        targets = [{k: v for k, v in t.items()} for t in targets]

        predictions = self.pytorch_model(images)
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
        optimizer = torch.optim.SGD(self.pytorch_model.parameters(), lr=0.001, weight_decay = 1e-3, momentum=0.9)
        return optimizer
    

# Función para crear el modelo con configuración
def create_model(num_classes, config):
    return Faster_VGG16_CustomModel(num_classes=num_classes, config=config)
