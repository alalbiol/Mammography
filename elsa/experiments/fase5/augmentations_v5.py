"""
experiments/fase5/augmentations_v5.py
======================================
Aumentado para E05_v5: augmentations básicas de Alberto + RandAugment simulado.

RandAugment no está disponible en albumentations 2.0.8, se simula con
RandomOrder que aplica n transformaciones aleatorias de la lista en cada imagen.
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
        # RandAugment simulado: selecciona 2 de estas transformaciones al azar
        A.RandomOrder([
            A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
            A.GaussNoise(std_range=(0.002, 0.01), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.RandomToneCurve(scale=0.05, p=1.0),
        ], n=2, p=0.5),
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
    return transforms.Compose([
        SafeStandardize(),
        RandomIntensity(0.8, 1.2),
        RandomContrast(0.8, 1.2),
    ])
