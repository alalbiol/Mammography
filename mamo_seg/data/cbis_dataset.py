
    
import pathlib
from PIL import Image
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.load_config import get_parameter


class  CBISSegmentationPatchDataset(Dataset):
    def __init__(self, dataset_root, split = None, crop_size = 224, transform=None):
        """
        Args:
            images_dir (str): Directory containing images.
            masks_dir (str): Directory containing masks.
            transform (callable, optional): Albumentations transformations for images and masks.
        """
        self.dataset_root = pathlib.Path(dataset_root)
        self.images_dir = self.dataset_root / 'images'
        self.masks_dir = self.dataset_root / 'masks'
        self.bounding_boxes = pd.read_csv(self.dataset_root / 'bounding_boxes.csv')
        
        if split is not None:
            assert split in ['train', 'test'], "Invalid split"
            self.bounding_boxes = self.bounding_boxes[self.bounding_boxes['group'] == split]
        
        self.transform = transform
        
        self.crop_size = crop_size

        # List all images and masks, ensuring corresponding pairs
        self.image_filenames = list(self.images_dir.glob('*.png'))
        self.mask_filenames = list(self.masks_dir.glob('*.png'))

        assert len(self.image_filenames) == len(self.mask_filenames), \
            "Mismatch in number of images and masks"

    def __len__(self):
        return len(self.bounding_boxes)

    def __getitem__(self, idx):
        # Get image and mask file paths for the given bounding box
        
        id = self.bounding_boxes.iloc[idx]['id']
        
        img_path = self.images_dir / f'{id}.png'
        mask_path = self.masks_dir / f'{id}.png'
        
        bb_center_x = self.bounding_boxes.iloc[idx]['x'] + self.bounding_boxes.iloc[idx]['w'] // 2
        bb_center_y = self.bounding_boxes.iloc[idx]['y'] + self.bounding_boxes.iloc[idx]['h'] // 2

        # Open images
        image = np.array(Image.open(img_path)).astype(np.float32)
        image = image / 65535.0  # Normalize to [0, 1]
        mask = np.array(Image.open(mask_path))
        
        # Crop image and mask around the bounding box center. If the bounding box is close to the border,
        # limit the crop to the image size 
        crop_min_x = max(0, bb_center_x - self.crop_size // 2)
        crop_min_y  = max(0, bb_center_y - self.crop_size // 2)
        crop_max_x = min(image.shape[1], crop_min_x + self.crop_size)
        crop_max_y = min(image.shape[0], crop_min_y + self.crop_size)
        
        if crop_max_x - crop_min_x < self.crop_size:
            crop_min_x = crop_max_x - self.crop_size
        if crop_max_y - crop_min_y < self.crop_size:
            crop_min_y = crop_max_y - self.crop_size
            
        assert crop_max_x - crop_min_x == self.crop_size, f"Invalid crop size: {crop_max_x - crop_min_x}"
        assert crop_max_y - crop_min_y == self.crop_size, f"Invalid crop size: {crop_max_y - crop_min_y}"
        
        crop_image = image[crop_min_y:crop_max_y, crop_min_x:crop_max_x]
        crop_mask = mask[crop_min_y:crop_max_y, crop_min_x:crop_max_x]
        
        
        

        # Apply transformations if specified
        if self.transform:
            augmented = self.transform(image=crop_image, mask=crop_mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask.long()



# Create data module
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
from utils.transforms import StandardizeImage

class CBISSegmentationDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dataset_root = get_parameter(self.config, ['DataModule', 'dataset_root'])
        
        
        

    def prepare_data(self):
        # Download or prepare the data if needed
        dataset_root = get_parameter(self.config, ['DataModule', 'dataset_root'])
        assert pathlib.Path(dataset_root).exists(), f"Data directory {self.ddsm_root} does not exist."
        
    
    
    def setup(self, stage=None):
        # Create train, val, test datasets
        pass # everythin is in get_train_dataloader and get_test_dataloader
    
        
    
    def setup(self, stage=None):
        pass
    
    def train_dataloader(self):
        
       
        
        
        crop_size = get_parameter(self.config, ['DataModule',  'crop_size'], default=512)
        
        
        batch_size = get_parameter(self.config, ['DataModule',  'batch_size'])
        num_workers = get_parameter(self.config, ['DataModule', 'num_workers'], default=16)
        

    
        transform = A.Compose([
            A.Resize(crop_size, crop_size),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(
                scale=(0.8, 1.2),  # Random scaling between 80% and 120%
                rotate=(-30, 30),  # Random rotation between -30 and 30 degrees
                shear=(-15, 15),   # Random shearing between -15 and 15 degrees
                p=1.0              # Probability of applying the transform
            ),
            A.RandomBrightnessContrast(p=0.2),
            StandardizeImage(mean=0.0, std=1.0, always_apply=True),
            ToTensorV2()
        ])
        
        dataset = CBISSegmentationPatchDataset(self.dataset_root, split='train', crop_size=crop_size, transform=transform)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    
    def val_dataloader(self):
        
        
        crop_size = get_parameter(self.config, ['DataModule',  'crop_size'], default=512)
        
        
        batch_size = get_parameter(self.config, ['DataModule',  'batch_size'])
        num_workers = get_parameter(self.config, ['DataModule', 'num_workers'], default=16)
        
        transform = A.Compose([
            A.Resize(crop_size, crop_size),
            StandardizeImage(mean=0.0, std=1.0, always_apply=True),
            ToTensorV2()
        ])
        
        dataset = CBISSegmentationPatchDataset(self.dataset_root, split='test', crop_size=crop_size, transform=transform)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    