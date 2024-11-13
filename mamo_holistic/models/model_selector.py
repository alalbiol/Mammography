import torch.nn as nn
from torchvision import models
from .nikulin import NikulinPatchModel


def get_patch_model(model_name, num_classes = 5,  **kwargs):
    """
    Get the model class based on the model name.

    Args:
        model_name (str): The name of the model.

    Returns:
        The model class.
    """
    if model_name == "nikulin":
        print("Using Nikulin model")
        return NikulinPatchModel(**kwargs)
    
    if model_name == "resnet18":
        print("Using ResNet18 model")
        model =  models.resnet18( weights=models.ResNet18_Weights.DEFAULT)
        # Modify the last layer to match the number of classes
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    
    if model_name == "resnet34":
        print("Using ResNet34 model")
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        # Modify the last layer to match the number of classes
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    
    if model_name == "resnet50":
        print("Using ResNet50 model")
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Modify the last layer to match the number of classes
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    
    
    
    raise ValueError(f"Model {model_name} not found.")