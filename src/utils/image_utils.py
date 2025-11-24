"""Image processing utilities for the Weighted Integrated Gradients library."""

import torch
import numpy as np
from PIL import Image
from typing import Tuple, Optional, List

from ..config import DEFAULT_MEAN, DEFAULT_STD


def normalize(image: np.ndarray, mean: Optional[List[float]] = None, std: Optional[List[float]] = None) -> torch.Tensor:
    """Normalize image with mean and standard deviation.
    
    Args:
        image: Input image array with shape (B, H, W, C) or (H, W, C)
        mean: Mean values for each channel. Defaults to ImageNet mean.
        std: Standard deviation values for each channel. Defaults to ImageNet std.
        
    Returns:
        Normalized image tensor with channels in PyTorch format (B, C, H, W)
    """
    if mean is None:
        mean = DEFAULT_MEAN
    if std is None:
        std = DEFAULT_STD
        
    if image.max() > 1:
        image = image.astype(np.float64)
        image /= 255
        
    image = (image - mean) / std
    # Roll axis to PyTorch format: (H, W, C) -> (C, H, W)
    return torch.tensor(image.swapaxes(-1, 1).swapaxes(2, 3)).float()


def get_sample_image(image_path: str, transform) -> torch.Tensor:
    """Load and transform an image from path.
    
    Args:
        image_path: Path to the image file
        transform: Torchvision transform to apply
        
    Returns:
        Transformed image tensor
    """
    image = Image.open(image_path)
    transformed_image = transform(image)
    return transformed_image


def get_sample_mask(mask_path: str, transform) -> torch.Tensor:
    """Load and transform a mask from path.
    
    Args:
        mask_path: Path to the mask file
        transform: Torchvision transform to apply
        
    Returns:
        Transformed mask tensor
    """
    mask = Image.open(mask_path)
    transformed_mask = transform(mask)
    return transformed_mask


def get_sample_data(
    image_ind: int, 
    images_path: List[str], 
    masks_path: List[str], 
    transform
) -> Tuple[str, torch.Tensor, torch.Tensor]:
    """Load image and mask data for a given index.
    
    Args:
        image_ind: Index of the image to load
        images_path: List of image paths
        masks_path: List of mask paths
        transform: Torchvision transform to apply
        
    Returns:
        Tuple of (image_name, transformed_image, transformed_mask)
    """
    image_path = images_path[image_ind]
    mask_path = masks_path[image_ind]
    image_raw_name = image_path.split("/")[-1].split(".")[0]

    image = get_sample_image(image_path, transform)
    mask = get_sample_mask(mask_path, transform)

    return image_raw_name, image, mask


def get_neutral_background(image: np.ndarray) -> np.ndarray:
    """Compute neutral background color from image corners.
    
    This function computes the average color from the four corners of the image
    (10% of image size from each corner) to use as a neutral background.
    
    Args:
        image: Input image array with shape (H, W, C)
        
    Returns:
        Average corner color with shape (1, 1, C)
    """
    height, width = image.shape[:2]
    corner_size = int(0.1 * height)
    
    # Extract corners
    top_left = image[:corner_size, :corner_size, :]
    top_right = image[:corner_size, -corner_size:, :]
    bottom_left = image[-corner_size:, :corner_size, :]
    bottom_right = image[-corner_size:, -corner_size:, :]
    
    # Compute averages
    average_top_left = np.mean(top_left, axis=(0, 1))
    average_top_right = np.mean(top_right, axis=(0, 1))
    average_bottom_left = np.mean(bottom_left, axis=(0, 1))
    average_bottom_right = np.mean(bottom_right, axis=(0, 1))
    
    # Overall average
    average_all_corners = np.mean(
        [average_top_left, average_top_right, average_bottom_left, average_bottom_right], 
        axis=0
    )
    
    # Broadcast to (1, 1, C) shape
    average_all_corners_broadcasted = average_all_corners[np.newaxis, np.newaxis, :]
    return average_all_corners_broadcasted


def convert_to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert RGB image to grayscale using standard weights.
    
    Args:
        img: RGB image array with shape (C, H, W) where C=3
        
    Returns:
        Grayscale image with shape (H, W)
    """
    result = 0.2989 * img[0, :, :] + 0.5870 * img[1, :, :] + 0.1140 * img[2, :, :]
    return result

