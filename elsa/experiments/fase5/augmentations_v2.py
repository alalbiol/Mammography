import torch
import albumentations as A
from torchvision import transforms
from utils.transforms import RandomContrast, RandomIntensity, Standardize


def get_geometric_transform_v2():
    return A.Compose([
        A.Affine(
            scale=(0.7, 1.3),
            shear=(-15, 15),
            rotate=(-180, 180),
            p=1.0
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ElasticTransform(alpha=50, sigma=10, p=0.5),
        A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.5
        ),
    ], additional_targets={'mask': 'mask'})


class SafeStandardize:
    """Standardize con protección contra std=0."""
    def __call__(self, img):
        mean = img.mean()
        std  = img.std()
        if std < 1e-6:
            std = torch.tensor(1.0)
        return (img - mean) / std


def get_intensity_transform_v2():
    return transforms.Compose([
        SafeStandardize(),
        RandomIntensity(0.6, 1.4),
        RandomContrast(0.6, 1.4),
    ])
