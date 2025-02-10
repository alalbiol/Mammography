import numpy as np
import albumentations as A

class StandardizeImage(A.ImageOnlyTransform):
    """
    Custom Albumentation transform to standardize pixel values of images
    while leaving masks unchanged.
    """
    def __init__(self, mean=0.0, std=1.0, always_apply=False, p=1.0):
        """
        Args:
            mean (float): Mean value for standardization.
            std (float): Standard deviation for standardization.
            always_apply (bool): Whether to always apply the transformation.
            p (float): Probability of applying the transformation.
        """
        super().__init__(always_apply=always_apply, p=p)
        self.mean = mean
        self.std = std

    def apply(self, image, **params):
        """
        Apply standardization to the image.
        Args:
            image (numpy.ndarray): Input image to be standardized.
        Returns:
            numpy.ndarray: Standardized image.
        """
        return (image - np.mean(image)) / (np.std(image) + 1e-8) * self.std + self.mean
