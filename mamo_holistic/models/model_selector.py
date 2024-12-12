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
    
    if model_name == "resnet50_bn":
        print("Using ResNet50 model")
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Modify the last layer to match the number of classes
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        bn = nn.BatchNorm2d(3)
        
        model = nn.Sequential(bn, model)
    
    if "swin" in model_name:
        import timm
        print("Using Swin Transformer model:", model_name)
        pretrained = kwargs.get("pretrained", True)
        image_size = kwargs.get("image_size", 224)
        model = timm.create_model(model_name, pretrained=pretrained)
        model.head.fc = nn.Linear(model.head.fc.in_features, num_classes)
            
        model.set_input_size(image_size)
             
        return model
    
    
    
    raise ValueError(f"Model {model_name} not found.")


def get_image_model(model_name, num_classes = 2,  **kwargs):
    """
    Get the model class based on the model name.

    Args:
        model_name (str): The name of the model.

    Returns:
        The model class.
    """
    if model_name == "nikulin":
        print("Using nikulin model")
        from .nikulin import NikulinImage
        model =  NikulinImage(**kwargs)
        if "ckpt_path" in kwargs:
            print("Loading weights from", kwargs["ckpt_path"])
            model.load_weights_from_tf(kwargs["ckpt_path"])
        
        return model
    
    if model_name == "swin_base_patch4_window7_224":
        import timm
        print("Using Swin Transformer model")
        pretrained = kwargs.get("pretrained", True)
        image_size = kwargs.get("image_size", (1152,896))
        model = timm.create_model(model_name, pretrained=pretrained)
        model.head.fc = nn.Linear(model.head.fc.in_features, num_classes)
            
        model.set_input_size(image_size)
        
        return model
    
    if model_name == "swin_tiny_patch4_window7_224":
        import timm
        print("Using Swin Transformer model")
        pretrained = kwargs.get("pretrained", True)
        image_size = kwargs.get("image_size", (1152,896))
        model = timm.create_model(model_name, pretrained=pretrained)
        model.head.fc = nn.Linear(model.head.fc.in_features, num_classes)
            
        model.set_input_size(image_size)
        
        return model
        
        
    
    
    
    raise ValueError(f"Model {model_name} not found.")