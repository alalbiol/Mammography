import torch
import albumentations as A
from torchvision import transforms
from utils.transforms import RandomContrast, RandomIntensity, Standardize


def get_geometric_transform_v3():
    return A.Compose([
        A.Affine(
            scale=(0.8, 1.2),
            shear=10,
            rotate=(-15, 15),
            p=1.0
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.GaussNoise(std_range=(0.002, 0.01), p=0.2),
    ], additional_targets={'mask': 'mask'})


class SafeStandardize:
    def __call__(self, img):
        mean = img.mean()
        std  = img.std()
        if std < 1e-6:
            std = torch.tensor(1.0)
        return (img - mean) / std


def get_intensity_transform_v3():
    return transforms.Compose([
        SafeStandardize(),
        RandomIntensity(0.8, 1.2),
        RandomContrast(0.8, 1.2),
    ])
