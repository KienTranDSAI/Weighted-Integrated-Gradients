"""Model loading utilities."""

import torch
from torchvision import models
from typing import Optional


AVAILABLE_MODELS = {
    'densenet121': 'DenseNet121',
    'mobilenet_v2': 'MobileNetV2',
    'resnet50': 'ResNet50',
    'resnet101': 'ResNet101',
    'vgg16': 'VGG16',
    'vgg19': 'VGG19',
}


def load_model(model_name: str, device: Optional[str] = None) -> torch.nn.Module:
    """Load a pretrained model by name.
    
    Args:
        model_name: Name of the model to load. Available models:
                   'densenet121', 'mobilenet_v2', 'resnet50', 'resnet101', 
                   'vgg16', 'vgg19'
        device: Device to load model on. If None, uses CPU.
        
    Returns:
        Loaded and evaluated PyTorch model
        
    Raises:
        ValueError: If model_name is not supported
    """
    model_name = model_name.lower()
    
    if model_name not in AVAILABLE_MODELS:
        available = ', '.join(AVAILABLE_MODELS.keys())
        raise ValueError(
            f"Model '{model_name}' is not supported. "
            f"Available models: {available}"
        )
    
    print(f"Loading model: {AVAILABLE_MODELS[model_name]}...")
    
    if model_name == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=True)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
    elif model_name == 'resnet101':
        model = models.resnet101(pretrained=True)
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif model_name == 'vgg19':
        model = models.vgg19(pretrained=True)
    
    model = model.eval()
    
    if device is not None:
        model = model.to(device)
    
    print(f"Model {AVAILABLE_MODELS[model_name]} loaded successfully.")
    return model


def get_available_models() -> list:
    """Get list of available model names.
    
    Returns:
        List of available model names
    """
    return list(AVAILABLE_MODELS.keys())

