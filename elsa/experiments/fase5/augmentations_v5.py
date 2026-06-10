"""
experiments/fase5/augmentations_v5.py
======================================
Aumentado para E05_v5: augmentations básicas de Alberto + RandAugment conservador.

MOTIVACIÓN
----------
v1_fixed usa las augmentations básicas del DDSMImageDataModule original
(Affine + HFlip + VFlip + RandomIntensity + RandomContrast).
Añadimos RandAugment con parámetros conservadores (n=2, magnitude=6)
para aumentar la variedad sin ser agresivos.
"""

import torch
import albumentations as A
from torchvision import transforms
from utils.transforms import RandomContrast, RandomIntensity, Standardize


def get_geometric_transform_v5():
    return A.Compose([
        # Augmentations básicas de Alberto (igual que v1_fixed)
        A.Affine(scale=(0.8, 1.2), shear=10, rotate=(-15, 15), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # RandAugment conservador: selecciona 2 augmentations aleatorias
        # con magnitud 6 (escala 0-30, conservador)
        A.RandAugment(n=2, magnitude=6, p=1.0),
    ], additional_targets={'mask': 'mask'})


class SafeStandardize:
    """Standardize con protección contra std=0."""
    def __call__(self, img):
        mean = img.mean()
        std  = img.std()
        if std < 1e-6:
            std = torch.tensor(1.0)
        return (img - mean) / std


def get_intensity_transform_v5():
    # Igual que v1_fixed
    return transforms.Compose([
        SafeStandardize(),
        RandomIntensity(0.8, 1.2),
        RandomContrast(0.8, 1.2),
    ])