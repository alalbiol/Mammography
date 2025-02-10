import segmentation_models_pytorch as smp


def create_unet_model(model_params):
    encoder_name = model_params["encoder_name"]
    encoder_weights = model_params.get("encoder_weights", "imagenet")
    in_channels = model_params["in_channels"]
    classes = model_params["num_classes"]
    attention_type = model_params.get("attention_type", None)
    
    print(in_channels)

    # Create the model with ResNet32 backbone and SE attention
    model = smp.Unet(
        encoder_name=encoder_name,  # Use ResNet32 as the encoder backbone
        encoder_weights= encoder_weights,  # Optionally use pre-trained weights
        in_channels=in_channels,  # Number of input channels (e.g., 3 for RGB images)
        classes=classes,  # Number of output classes (5 classes for segmentation)
        decoder_attention_type=attention_type  # Enable SE attention
    )

    return model



def get_segmentation_model(model_name, model_params):
    
    if model_name.lower() == "unet":
        model = create_unet_model(model_params)
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    return model