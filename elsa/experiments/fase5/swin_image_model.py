"""
experiments/fase5/swin_image_model.py
======================================
Fase V — Modelo para clasificación binaria de mamografías completas.

ARQUITECTURA CORREGIDA (indicación de Alberto):
  Tras el AvgPool2d el mapa (B, 1536, 5, 4) pasa por el patch_classifier
  Conv1x1 (1536→5) → (B, 5, 5, 4), y luego se hace flatten a (B, 100).
  A partir de ahí las capas son Linear (FF), no Conv2d.
  La conexión residual va de 100→2 directamente.

Flujo completo:
  Imagen (B,1,H,W)
    → Gray2RGB (B,3,H,W)
    → Swin Large forward_features (B,35,28,1536)
    → AvgPool2d(7) (B,1536,5,4)
    → patch_classifier Conv1x1 (B,5,5,4)
    → Flatten (B,100)
    → [Camino principal]  LayerNorm+Linear 100→16→8→2  (+Dropout 0.3)
    → [Shortcut]          Linear 100→2
    → Suma → 2 logits

CAMBIOS v3:
  - Dropout(0.3) entre cada capa FF del camino principal para reducir overfitting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ─────────────────────────────────────────────────────────────────────────────
# Gray → RGB
# ─────────────────────────────────────────────────────────────────────────────

class Gray2RGBadaptor(nn.Module):
    """Replica el canal gris 3 veces para alimentar modelos RGB."""
    def forward(self, x):
        return x.repeat(1, 3, 1, 1)


# ─────────────────────────────────────────────────────────────────────────────
# NikulinFusion — versión corregida con capas FF (Linear)
# ─────────────────────────────────────────────────────────────────────────────

class NikulinFusion(nn.Module):
    """
    Cabeza de clasificación de imagen completa siguiendo a Nikulin.

    Entrada: mapa de características (B, H, W, C) = (B, 5, 4, 1536)
    Salida : 2 logits (no cáncer / cáncer)

    Parámetros:
        input_feat      : dimensión de los tokens del Swin (1536 para Large)
        kernel_size_x   : anchura del mapa tras AvgPool2d(7) → 896/32/7 = 4
        kernel_size_y   : altura  del mapa tras AvgPool2d(7) → 1120/32/7 = 5
        num_patch_classes: clases del modelo de Fase IV (5)
        dropout         : probabilidad de dropout entre capas FF (default: 0.3)
    """

    def __init__(self,
                 input_feat: int = 1536,
                 kernel_size_x: int = 4,
                 kernel_size_y: int = 5,
                 num_patch_classes: int = 5,
                 dropout: float = 0.3):
        super().__init__()
        self.input_feat        = input_feat
        self.kernel_size_x     = kernel_size_x
        self.kernel_size_y     = kernel_size_y
        self.num_patch_classes = num_patch_classes

        # Número de zonas tras AvgPool y número de valores tras patch_classifier
        n_zonas  = kernel_size_x * kernel_size_y          # 4×5 = 20
        n_flat   = n_zonas * num_patch_classes             # 20×5 = 100

        # Clasificador de zona: Conv1×1 proyecta 1536 → 5 por zona
        # Sus pesos se inicializan desde la cabeza del Swin de Fase IV
        self.patch_classifier = nn.Conv2d(
            input_feat, num_patch_classes,
            kernel_size=1, stride=1, padding=0, bias=True
        )

        # Camino principal: capas FF (Linear) con LayerNorm y Dropout
        # 100 → 16 → 8 → 2
        self.ln1   = nn.LayerNorm(n_flat)
        self.drop1 = nn.Dropout(dropout)
        self.fc1   = nn.Linear(n_flat, 16, bias=False)

        self.ln2   = nn.LayerNorm(16)
        self.drop2 = nn.Dropout(dropout)
        self.fc2   = nn.Linear(16, 8, bias=False)

        self.ln3   = nn.LayerNorm(8)
        self.drop3 = nn.Dropout(dropout)
        self.fc3   = nn.Linear(8, 2, bias=False)

        # Conexión residual directa: 100 → 2
        self.shortcut = nn.Linear(n_flat, 2, bias=False)

    def init_from_patchmodel(self, patch_model):
        """
        Inicializa patch_classifier con los pesos de la cabeza del Swin de Fase IV.
        La cabeza del Swin es: head.fc = Sequential(Dropout, Linear(1536→5))
        """
        weights = patch_model[1].head.fc[1].weight[:5, :].data.cpu()  # (5, 1536)
        bias    = patch_model[1].head.fc[1].bias[:5].data.cpu()        # (5,)
        self.patch_classifier.weight.data = weights.reshape(5, self.input_feat, 1, 1)
        self.patch_classifier.bias.data   = bias
        print("[NikulinFusion] patch_classifier inicializado desde checkpoint Fase IV")

    def forward(self, x):
        # x: (B, H, W, C) = (B, 5, 4, 1536)

        # Reordenamos para Conv2d: (B, C, H, W)
        x_in = x.permute(0, 3, 1, 2).contiguous()   # (B, 1536, 5, 4)

        # Clasificador de zona: 1536 → 5 por zona
        x_in = self.patch_classifier(x_in)            # (B, 5, 5, 4)

        # Flatten espacial: (B, 5, 5, 4) → (B, 100)
        x_flat = x_in.flatten(start_dim=1)            # (B, 100)

        # Camino principal: LayerNorm + Dropout + Linear sucesivos
        x = F.relu(self.ln1(x_flat))
        x = self.drop1(x)
        x = self.fc1(x)                               # (B, 16)

        x = F.relu(self.ln2(x))
        x = self.drop2(x)
        x = self.fc2(x)                               # (B, 8)

        x = F.relu(self.ln3(x))
        x = self.drop3(x)
        x = self.fc3(x)                               # (B, 2)

        # Conexión residual: 100 → 2
        shortcut = self.shortcut(x_flat)              # (B, 2)

        # Suma camino principal + shortcut
        return x + shortcut                           # (B, 2)


# ─────────────────────────────────────────────────────────────────────────────
# SwinBreastCancerLarge — modelo completo
# ─────────────────────────────────────────────────────────────────────────────

class SwinBreastCancerLarge(nn.Module):
    """
    Swin Large (Fase IV) + NikulinFusion para clasificación binaria
    de mamografías completas (896×1120).

    Parámetros (config_fase5.yaml → model_params):
        patch_checkpoint  : ruta al mejor .ckpt de Fase IV
        freeze_patch_model: True = congela el Swin (recomendado)
        unfreeze_layer    : capa del Swin a descongelar (default: 3)
        image_size        : [W, H] de las imágenes (default: [896, 1120])
        dropout           : dropout en NikulinFusion (default: 0.3)
    """

    SWIN_MODEL_NAME = "swin_large_patch4_window7_224.ms_in22k"
    FEATURE_DIM     = 1536
    NUM_PATCH_CLS   = 5

    def __init__(self, num_classes: int = 2, **kwargs):
        super().__init__()

        image_size       = kwargs.get("image_size", (896, 1120))
        patch_checkpoint = kwargs.get("patch_checkpoint", None)
        freeze           = kwargs.get("freeze_patch_model", True)
        unfreeze_layer   = kwargs.get("unfreeze_layer", 3)
        dropout          = kwargs.get("dropout", 0.3)

        # Swin Large sin pesos ImageNet — cargamos los de Fase IV
        swin = timm.create_model(self.SWIN_MODEL_NAME, pretrained=False)
        swin.head.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(swin.head.fc.in_features, self.NUM_PATCH_CLS)
        )
        self.patch_model = nn.Sequential(Gray2RGBadaptor(), swin)

        if patch_checkpoint is not None:
            print(f"[SwinBreastCancerLarge] Cargando checkpoint: {patch_checkpoint}")
            ckpt = torch.load(patch_checkpoint, map_location="cpu")
            new_sd = {k.replace("model.", "", 1): v
                      for k, v in ckpt["state_dict"].items()
                      if k.startswith("model.")}
            missing, unexpected = self.patch_model[1].load_state_dict(new_sd, strict=False)
            print(f"  Missing keys: {len(missing)}  |  Unexpected: {len(unexpected)}")
        else:
            print("[SwinBreastCancerLarge] Sin checkpoint — pesos aleatorios")

        # Ajuste de resolución para imagen completa
        self.patch_model[1].set_input_size((image_size[1], image_size[0]))  # (H, W)

        # AvgPool2d(7): 35×28 → 5×4 = 20 zonas
        self.pooling = nn.AvgPool2d(7, stride=7)

        # NikulinFusion
        self.fusion = NikulinFusion(
            input_feat        = self.FEATURE_DIM,
            kernel_size_x     = 4,
            kernel_size_y     = 5,
            num_patch_classes = self.NUM_PATCH_CLS,
            dropout           = dropout,
        )
        if patch_checkpoint is not None:
            self.fusion.init_from_patchmodel(self.patch_model)

        # Congelar Swin excepto layer indicada + NikulinFusion
        if freeze:
            self._freeze_patch_model()
            if unfreeze_layer is not None:
                self._unfreeze_layer(unfreeze_layer)
                print(f"[SwinBreastCancerLarge] Swin congelado — layer {unfreeze_layer} + NikulinFusion entrenan")
            else:
                print("[SwinBreastCancerLarge] Swin completamente congelado — solo NikulinFusion entrena")

    def _freeze_patch_model(self):
        for param in self.patch_model.parameters():
            param.requires_grad = False

    def _unfreeze_layer(self, layer_idx):
        for param in self.patch_model[1].layers[layer_idx].parameters():
            param.requires_grad = True

    def forward(self, x):
        # x: (B, 1, 1120, 896)
        x_rgb           = self.patch_model[0](x)                       # (B, 3, 1120, 896)
        patch_features  = self.patch_model[1].forward_features(x_rgb)  # (B, 35, 28, 1536)
        patch_features  = patch_features.permute(0, 3, 1, 2)           # (B, 1536, 35, 28)
        pooled_features = self.pooling(patch_features)                  # (B, 1536, 5, 4)
        pooled_features = pooled_features.permute(0, 2, 3, 1)          # (B, 5, 4, 1536)
        return self.fusion(pooled_features)                             # (B, 2)