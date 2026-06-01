"""
data/sliding_window_dataset.py
==============================
Dataset para la Fase V: extrae parches de mamografías completas
mediante una ventana deslizante, sin augmentation (modo inferencia).

Cada __getitem__ devuelve TODOS los parches de una imagen junto con
su etiqueta binaria (0=normal, 1=cáncer) y el image_id, para que
eval_fusion.py pueda agregar las predicciones por imagen.

Uso esperado:
    dataset = SlidingWindowImageDataset(
        val_csv   = "resources/ddsm/DDSM_val.csv",
        annot_gz  = "resources/ddsm/ddsm_annotations_16bits_1120_896.json.gz",
        root_dir  = "/home/Data/mamo/DDSM_png_16bit_1120x896",
        patch_size = 224,
        stride     = 112,          # solapamiento 50 %
        convert_to_rgb = True,
        normalize_input = True,
    )
    # Usar con batch_size=1 y el collate_fn incluido
    loader = DataLoader(dataset, batch_size=1, collate_fn=sliding_window_collate_fn)
"""

import pathlib
import gzip
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _load_annotations(split_csv: str, annotations_file: str) -> pd.DataFrame:
    """
    Réplica mínima de la lógica de DDSM_Image_Dataset.load_annotations:
    une el CSV de split con el fichero de anotaciones y devuelve un
    DataFrame con columnas [image_id, breast_malignant].
    """
    split_images = pd.read_csv(split_csv)

    if str(annotations_file).endswith('.json'):
        annotations = pd.read_json(annotations_file, orient='records', lines=True)
    else:
        with gzip.open(annotations_file, 'rt', encoding='utf-8') as f:
            annotations = pd.read_json(f, orient='records', lines=True)

    records = []
    for image_id in split_images['ddsm_image']:
        if image_id in annotations['image_id'].values:
            malignant = annotations.loc[
                annotations['image_id'] == image_id, 'breast_malignant'
            ].values[0]
            records.append({'image_id': image_id, 'breast_malignant': bool(malignant)})
        elif 'normal' in image_id:
            records.append({'image_id': image_id, 'breast_malignant': False})
        # imágenes sin anotación (otro pecho en carpeta cancer/benign) → se descartan

    return pd.DataFrame(records)


def _extract_patches(image: np.ndarray,
                     patch_size: int,
                     stride: int) -> np.ndarray:
    """
    Extrae parches con ventana deslizante sobre una imagen 2-D (H, W).
    Los parches que sobresalen del borde se completan con ceros (padding).

    Returns:
        patches: np.ndarray (N, patch_size, patch_size) float32
    """
    H, W = image.shape
    patches = []

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            patch = np.zeros((patch_size, patch_size), dtype=np.float32)
            y_end = min(y + patch_size, H)
            x_end = min(x + patch_size, W)
            patch[:y_end - y, :x_end - x] = image[y:y_end, x:x_end]
            patches.append(patch)

    return np.stack(patches, axis=0)   # (N, patch_size, patch_size)


# ──────────────────────────────────────────────────────────────
# Dataset principal
# ──────────────────────────────────────────────────────────────

class SlidingWindowImageDataset(Dataset):
    """
    Devuelve, por cada mamografía, un tensor con todos sus parches
    extraídos por sliding window.

    Args:
        val_csv         : CSV del split (columna 'ddsm_image').
        annot_gz        : Fichero de anotaciones (.json o .json.gz).
        root_dir        : Directorio raíz de las imágenes DDSM.
        patch_size      : Tamaño del parche cuadrado (px).
        stride          : Paso del sliding window (px).  stride < patch_size → solapamiento.
        convert_to_rgb  : Si True, replica el canal gris 3 veces → (3, H, W).
        normalize_input : Si True, normaliza cada parche a media 0 / std 1.
        subset_size     : Si no es None, limita el dataset a ese nº de imágenes.
    """

    def __init__(self,
                 val_csv: str,
                 annot_gz: str,
                 root_dir: str,
                 patch_size: int = 224,
                 stride: int = 112,
                 convert_to_rgb: bool = True,
                 normalize_input: bool = True,
                 subset_size: int = None):

        self.root_dir       = pathlib.Path(root_dir)
        self.patch_size     = patch_size
        self.stride         = stride
        self.convert_to_rgb = convert_to_rgb
        self.normalize_input = normalize_input

        self.annotations = _load_annotations(val_csv, annot_gz)

        if subset_size is not None:
            self.annotations = self.annotations.sample(
                min(subset_size, len(self.annotations)), random_state=42
            ).reset_index(drop=True)

        print(f"[SlidingWindowImageDataset] {len(self.annotations)} imágenes  |  "
              f"patch_size={patch_size}  stride={stride}  rgb={convert_to_rgb}")

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int):
        row      = self.annotations.iloc[idx]
        image_id = row['image_id']
        label    = int(row['breast_malignant'])   # 0 = normal, 1 = cáncer

        # ── Carga de imagen ──────────────────────────────────────
        image_path = self.root_dir / image_id
        image = np.array(Image.open(image_path)).astype(np.float32)

        # ── Sliding window ───────────────────────────────────────
        patches = _extract_patches(image, self.patch_size, self.stride)
        # patches: (N, patch_size, patch_size)

        # ── Normalización por parche ──────────────────────────────
        if self.normalize_input:
            mean = patches.mean(axis=(1, 2), keepdims=True)
            std  = patches.std(axis=(1, 2), keepdims=True)
            std[std == 0] = 1.0
            patches = (patches - mean) / std

        # ── Canal: gris → RGB ────────────────────────────────────
        if self.convert_to_rgb:
            patches = np.stack([patches, patches, patches], axis=1)  # (N, 3, P, P)
        else:
            patches = patches[:, np.newaxis, :, :]                   # (N, 1, P, P)

        patches_tensor = torch.from_numpy(patches).float()   # (N, C, P, P)

        return patches_tensor, label, image_id


# ──────────────────────────────────────────────────────────────
# Collate fn  (usar con batch_size=1)
# ──────────────────────────────────────────────────────────────

def sliding_window_collate_fn(batch):
    """
    Collate function para usar con batch_size=1.
    Devuelve:
        patches : (N, C, P, P)   tensor de parches de UNA imagen
        label   : int
        image_id: str
    """
    assert len(batch) == 1, "SlidingWindowImageDataset debe usarse con batch_size=1"
    patches, label, image_id = batch[0]
    return patches, label, image_id