"""experiments/fase5/swin_image_model.py
=====================================

Implementación del modelo para la Fase V. 

En la Fase IV entrené un Swin Transformer para clasificar PARCHES de 224x224
en 5 clases (normal, masa benigna, calc benigna, masa maligna, calc maligna).
 
Ahora, siguiendo la metodología de Nikulin, queremos reutilizar ese modelo
para clasificar IMÁGENES COMPLETAS (896x1120) en 2 clases: cáncer / no cáncer.

Para hacerlo, usamos el Swin como extractor de características que procesa la imagen
completa produciendo un MAPA de características de tamaño 5x4 (una predicción por zona).
Encima de ese mapa añadimos una cabeza nueva (NikulinFusion) que aprende a combinar
todas las zonas y decidir si la mama es cancerosa o no. 

OJO! Al pasar de parches a imágenes completas tenemos muchas menos muestras de entrenamiento. 
Para evitar overfitting:
  - Congelamos TODOS los pesos del Swin (que ya aprendió a detectar lesiones).
  - Solo dejamos libres la última capa del Swin y la NikulinFusion.
  - Así el modelo solo tiene que aprender a COMBINAR las características y no a extraerlas.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ─────────────────────────────────────────────────────────────────────────────
# Gris → RGB 
# ─────────────────────────────────────────────────────────────────────────────

class Gray2RGBadaptor(nn.Module):
    """Adaptador que convierte imágenes en escala de grises a RGB.

    Motivo: Swin espera 3 canales y en mamografía trabajamos con imágenes de un solo canal, 
    así que para poder reutilizar los pesos preentrenados se ha de replicar el canal 3 veces.

    Entrada: tensor x = (B, 1, H, W)
    Salida: tensor con forma (B, 3, H, W)
    """
    def forward(self, x):
        return x.repeat(1, 3, 1, 1)


# ─────────────────────────────────────────────────────────────────────────────
# NikulinFusion  (para mapa 5x4 zonas con 1536 caracteristicas)
# ─────────────────────────────────────────────────────────────────────────────

class NikulinFusion(nn.Module):
    """
    Fusión estilo Nikulin sobre el mapa de características del Swin.
    input_feat : dimensión de los tokens del Swin (1536 para Large)
    kernel_size_x : anchura del mapa de características tras AvgPool2d(7)  → 896/32/7 = 4
    kernel_size_y : altura  del mapa de características tras AvgPool2d(7)  → 1120/32/7 = 5

    Entrada: Mapa de características del Swin de tamaño (B, 5, 4, C)
    Salida: 2 logits / probabilidad de normal y probabilidad de cáncer.

    """
    def __init__(self, input_feat: int = 1536,
                 kernel_size_x: int = 4,
                 kernel_size_y: int = 5,
                 num_patch_classes: int = 5):
        super().__init__()
        self.input_feat = input_feat
        kernel_size = (kernel_size_y, kernel_size_x)

        # Clasificador de parche: proyecta características → 5 scores (hereda del Fase IV)
        # Sus pesos se inicializan desde la cabeza del Swin de Fase IV
        self.patch_classifier = nn.Conv2d(input_feat, num_patch_classes,
                                          kernel_size=1, stride=1, padding=0, bias=True)

        # Cabeza de fusión: toma el mapa 5x4 de scores y lo colapsa a 2 logits
        self.ln1_final = nn.LayerNorm([num_patch_classes, kernel_size_y, kernel_size_x])
        self.fc1_final = nn.Conv2d(num_patch_classes, 16,
                                   kernel_size=kernel_size, stride=1, padding=0, bias=False)

        self.ln2_final = nn.LayerNorm([16, 1, 1])
        self.fc2_final = nn.Conv2d(16, 8, kernel_size=1, stride=1, padding=0, bias=False)

        self.ln3_final = nn.LayerNorm([8, 1, 1])
        self.fc3_final = nn.Conv2d(8, 2, kernel_size=1, stride=1, padding=0, bias=False)

        # Conexión residua
        self.ln_shortcut = nn.LayerNorm([num_patch_classes, kernel_size_y, kernel_size_x])
        self.shortcut_final = nn.Conv2d(num_patch_classes, 2,
                                        kernel_size=kernel_size, stride=1, padding=0, bias=False)

    def init_from_patchmodel(self, patch_model):
        """
        Inicializa patch_classifier con los pesos de la cabeza del Swin de Fase IV.
        patch_model es el nn.Sequential(Gray2RGBadaptor, swin).
        La cabeza del Swin es: head.fc = Sequential(Dropout, Linear(1536→5))
        """
        # Extraemos los pesos de la última Linear del Swin usada en Fase IV.
        # head.fc = [Dropout, Linear(in_features=1536, out_features=5)]
        # Tomamos la matriz de pesos (5, 1536) y el bias (5,)
        weights = patch_model[1].head.fc[1].weight[:5, :].data.cpu()  # (5, 1536)
        bias    = patch_model[1].head.fc[1].bias[:5].data.cpu()        # (5,)
        self.patch_classifier.weight.data = weights.reshape(5, self.input_feat, 1, 1)
        self.patch_classifier.bias.data   = bias
        print("[NikulinFusion] patch_classifier inicializado desde checkpoint Fase IV")

    def forward(self, x):
        # Entrada: x = (B, H, W, C) = (B, 5, 4, 1536) mapa de características del Swin 
        # H/W son las dimensiones espaciales de los tokens y C == input_feat (1536).
        
        # Reordenamos para que las convoluciones trabajen en (B, C, H, W)
        x_in = x.permute(0, 3, 1, 2).contiguous()   

        # Clasificamos cada zona: 1536 features → 5 scores de patología
        x_in = self.patch_classifier(x_in)  # (B, 5, 5, 4)

        # Bloque de reducción y clasificación final: cancer / no cancer
        x = F.relu(self.ln1_final(x_in))
        x = self.fc1_final(x)                         # (B, 16, 1, 1)

        x = F.relu(self.ln2_final(x))
        x = self.fc2_final(x)                         # (B, 8, 1, 1)

        x = F.relu(self.ln3_final(x))
        x = self.fc3_final(x)                         # (B, 2, 1, 1)

        # Conexión residual
        shortcut = self.ln_shortcut(x_in)
        shortcut = self.shortcut_final(shortcut)      # (B, 2, 1, 1)

        # Suma camino principal + shortcut → logits finales
        return (x + shortcut)[:, :, 0, 0]            # (B, 2)


# ─────────────────────────────────────────────────────────────────────────────
# MODELO COMPLETO: SwinBreastCancerLarge
#
# Encadena los tres pasos:
#   PASO 0: Gray → RGB
#   PASO 1: Swin Large (extractor de características preentrenado en Fase IV)
#   PASO 2: NikulinFusion (cabeza de clasificación, NUEVO)
# ─────────────────────────────────────────────────────────────────────────────
 
class SwinBreastCancerLarge(nn.Module):
    """
    Swin Large entrenado en Fase IV + NikulinFusion para clasificación binaria
    de mamografías completas (896x1120).

    Parámetros (vienen del config_fase5.yaml → model_params):
        patch_checkpoint : ruta al mejor .ckpt de Fase IV  (DDSMPatchClassifier)
        freeze_patch_model: True/False (default: True)
        unfreeze_layer   : índice de la capa Swin a descongelar (default: 3, la última)
        image_size       : [W, H] de las imágenes completas (default: [896, 1120])
    """

    SWIN_MODEL_NAME = "swin_large_patch4_window7_224.ms_in22k" ### IMMMMMP
    FEATURE_DIM     = 1536 # dimensión de los tokens del Swin Larg
    NUM_PATCH_CLS   = 5    # número de clases de parche (normal, masa benigna, calc benigna, masa maligna, calc maligna)

    def __init__(self, num_classes: int = 2, **kwargs):
        super().__init__()

        image_size       = kwargs.get("image_size", (896, 1120))   # (W, H)
        patch_checkpoint = kwargs.get("patch_checkpoint", None)
        freeze           = kwargs.get("freeze_patch_model", True)
        unfreeze_layer   = kwargs.get("unfreeze_layer", 3)

        # ── PASO 0 + PASO 1: Gray2RGB + Swin Large ───────────────────────
        # Creamos el Swin Large sin pesos preentrenados de ImageNet (pretrained=False)
        # porque vamos a cargar los pesos de Fase IV a continuación

        swin = timm.create_model(self.SWIN_MODEL_NAME, pretrained=False)
        swin.head.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(swin.head.fc.in_features, self.NUM_PATCH_CLS)
        )

        self.patch_model = nn.Sequential(Gray2RGBadaptor(), swin)

        # Carga pesos del mejor modelo de la Fase IV (formato DDSMPatchClassifier: prefijo 'model.')
        if patch_checkpoint is not None:
            print(f"[SwinBreastCancerLarge] Cargando checkpoint: {patch_checkpoint}")
            ckpt       = torch.load(patch_checkpoint, map_location="cpu")
            state_dict = ckpt["state_dict"]
            # Quitar prefijo 'model.' que añade DDSMPatchClassifier
            new_sd = {k.replace("model.", "", 1): v
                      for k, v in state_dict.items()
                      if k.startswith("model.")}
            # patch_model = Sequential(Gray2RGBadaptor[0], swin[1])
            # Los pesos del Swin en Fase IV no tienen el Gray2RGBadaptor,
            # así que los cargamos directamente en patch_model[1]
            missing, unexpected = self.patch_model[1].load_state_dict(new_sd, strict=False)
            print(f"  Missing keys : {len(missing)}")
            print(f"  Unexpected   : {len(unexpected)}")
        else:
            print("[SwinBreastCancerLarge] Sin checkpoint — pesos aleatorios")

        # Le decimos al Swin que ahora va a procesar imágenes completas de 1120x896 y no parches de 224x224. Esto es importante porque el Swin tiene embeddings posicionales
        # Internamente ajusta los embeddings posicionales mediante interpolación.
        swin_input_h = image_size[1] // 2   # 1120 // 2 = 560  (el Gray2RGB no cambia resolución)
        swin_input_w = image_size[0] // 2   # 896  // 2 = 448
        # Nota: dividO entre 2 porque Alberto lo hace así en base_config_image_swin
        # Si tus imágenes ya están a la resolución final, usa image_size directamente:
        swin_input_h = image_size[1]         # 1120
        swin_input_w = image_size[0]         # 896
        self.patch_model[1].set_input_size((swin_input_h, swin_input_w))

        # AvgPool2d(7) para reducir el mapa de características 
        # Swin divide por 32 → 1120/32=35, 896/32=28
        # Tras AvgPool2d(7): 35/7=5 (H), 28/7=4 (W)
        # Resultado: 20 zonas de la imagen, cada una con 1536 features
        self.pooling = nn.AvgPool2d(7, stride=7)

        # ── PASO 2: NikulinFusion ──────────────────────────────────────────
        self.fusion = NikulinFusion(
            input_feat      = self.FEATURE_DIM,
            kernel_size_x   = 4,   # W: 896/32/7 = 4
            kernel_size_y   = 5,   # H: 1120/32/7 = 5
            num_patch_classes = self.NUM_PATCH_CLS,
        )
        # Inicializamos la cabeza de fusión con los pesos del clasificador de Fase IV
        if patch_checkpoint is not None:
            self.fusion.init_from_patchmodel(self.patch_model)

        # ── Congelar / descongelar ────────────────────────────────────────
        if freeze:
            self._freeze_patch_model()
            self._unfreeze_layer(unfreeze_layer)
            # Informar al usuario qué capas quedan entrenables
            print(f"[SwinBreastCancerLarge] patch_model congelado, layer {unfreeze_layer} descongelada")

    def _freeze_patch_model(self):
        """Congela todos los pesos del Swin."""
        for param in self.patch_model.parameters():
            param.requires_grad = False

    def _unfreeze_layer(self, layer_idx):
        """Descongela una capa específica del Swin (layer 3 = última)."""
        for param in self.patch_model[1].layers[layer_idx].parameters():
            param.requires_grad = True

    def forward(self, x):
        """
        Paso completo de inferencia para una imagen completa.
        x: (B, 1, 1120, 896) — batch de mamografías en escala de grises
        """

        # PASO 0: gris → RGB
        x_rgb = self.patch_model[0](x)                      # (B, 3, H, W)

        # PASO 1: Swin extrae las características espaciales 
        patch_features = self.patch_model[1].forward_features(x_rgb) # (B, 35, 28, 1536)
        patch_features = patch_features.permute(0, 3, 1, 2)          # (B, 1536, 35, 28)

        # Reducimos el mapa espacialmente con AvgPool
        pooled_features = self.pooling(patch_features)              # (B, 1536, 5, 4)
        pooled_features = pooled_features.permute(0, 2, 3, 1)       # (B, 5, 4, 1536)

        # PASO 2: NikulinFusion combina las 20 zonas y decide cáncer/no cáncer
        return self.fusion(pooled_features)                         # (B, 2)