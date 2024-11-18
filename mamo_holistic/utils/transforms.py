import torch
from torchvision import transforms

class RandomContrast(object):
    def __init__(self, min_factor=0.8, max_factor=1.2):
        self.min_factor = min_factor
        self.max_factor = max_factor

    def __call__(self, img):
        factor = torch.empty(1).uniform_(self.min_factor, self.max_factor).item()
        return transforms.functional.adjust_contrast(img, factor)
    
class RandomIntensity(object):
    def __init__(self, min_factor=0.8, max_factor=1.2):
        self.min_factor = min_factor
        self.max_factor = max_factor

    def __call__(self, img):
        factor = torch.empty(1).uniform_(self.min_factor, self.max_factor).item()
        return transforms.functional.adjust_brightness(img, factor)