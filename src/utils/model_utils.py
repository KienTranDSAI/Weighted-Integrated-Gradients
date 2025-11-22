"""Model inference utilities for the Weighted Integrated Gradients library."""

import torch
import numpy as np
from typing import Optional, List, Union


def get_true_id(img: np.ndarray, model: torch.nn.Module, device: str = 'cpu') -> int:
    """Get the predicted class ID for an image.
    
    Args:
        img: Input image array with shape (B, H, W, C)
        model: PyTorch model for inference
        device: Device to run inference on
        
    Returns:
        Predicted class ID (integer)
    """
    model.to(device)
    img_tensor = torch.from_numpy(img).permute(0, 3, 1, 2).to(device)
    y_pred = model(img_tensor)
    class_id = torch.argmax(y_pred, dim=1)
    return class_id.item()


def get_score(
    model: torch.nn.Module, 
    device: str, 
    image: np.ndarray, 
    img_indice: Optional[int] = None
) -> float:
    """Get model confidence score for a specific class.
    
    Args:
        model: PyTorch model for inference
        device: Device to run inference on
        image: Input image array with shape (B, H, W, C)
        img_indice: Target class index. If None, uses predicted class.
        
    Returns:
        Softmax probability score for the target class
    """
    image_tensor = torch.from_numpy(image).permute(0, 3, 1, 2)
    model.to(device)
    image_tensor = image_tensor.to(device)
    scores = model(image_tensor)
    
    if img_indice is None:
        img_indice = torch.argmax(scores, dim=1).tolist()
        
    softmax_scores = torch.nn.functional.softmax(scores, dim=1)
    return softmax_scores[0, img_indice].item()


def get_partial_score_batch(
    images: List[np.ndarray], 
    model: torch.nn.Module, 
    device: str, 
    img_indices: Optional[Union[int, List[int]]] = None
) -> List[float]:
    """Get model confidence scores for a batch of images.
    
    Args:
        images: List of image arrays, each with shape (H, W, C)
        model: PyTorch model for inference
        device: Device to run inference on
        img_indices: Target class indices. If None, uses predicted classes.
                    Can be a single int (applied to all images) or a list.
        
    Returns:
        List of softmax probability scores for each image
    """
    batch_images = torch.from_numpy(np.array(images)).permute(0, 3, 1, 2)
    model.to(device)
    batch_images = batch_images.to(device)
    scores = model(batch_images)

    if img_indices is None:
        img_indices = torch.argmax(scores, dim=1).tolist()
    elif isinstance(img_indices, int):
        img_indices = [img_indices] * len(scores)

    softmax_scores = torch.nn.functional.softmax(scores, dim=1)
    softmax_scores_list = softmax_scores[torch.arange(len(img_indices)), img_indices].tolist()
    
    return softmax_scores_list

