"""Deletion metrics for evaluating attribution quality."""

import torch
import numpy as np
from typing import Optional, List

from .attribution_utils import get_sorted_indices, create_mask_from_indices
from ..utils.model_utils import get_score, get_partial_score_batch


def exact_find_d_alpha(
    model: torch.nn.Module,
    device: str,
    to_explain: np.ndarray,
    val: np.ndarray,
    trueImageInd: Optional[int] = None,
    target_ratio: float = 0.5,
    neutral_val: np.ndarray = 0,
    epsilon: float = 0.005,
    max_iter: int = 100
) -> tuple:
    """Find the number of pixels to delete to reach a target score ratio using binary search.
    
    This function uses binary search to find the optimal number of most important
    pixels that need to be deleted (replaced with neutral value) such that the
    model's confidence drops to approximately target_ratio * original_score.
    
    Args:
        model: PyTorch model for inference
        device: Device to run inference on
        to_explain: Original image array with shape (B, H, W, C)
        val: Importance values with shape (H, W)
        trueImageInd: Target class index. If None, uses predicted class.
        target_ratio: Target score as ratio of original score (default 0.5)
        neutral_val: Value to replace deleted pixels with
        epsilon: Convergence threshold for score difference
        max_iter: Maximum number of binary search iterations
        
    Returns:
        Tuple of (number of pixels deleted, final score)
    """
    low, high = 0, val.shape[0] * val.shape[1]
    sorted_indices = get_sorted_indices(val)
    full_score = get_score(model, device, to_explain, trueImageInd)
    target_score = full_score * target_ratio
    
    iter_count = 0
    score = full_score
    
    while high - low > 0 and iter_count < max_iter:
        iter_count += 1
        mid = int((low + high) / 2)
        
        # Create mask for top-mid most important pixels
        mask = create_mask_from_indices(val, mid, sorted_indices)
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=-1)
        background_mask = 1 - mask
        
        # Create partial image with deleted pixels
        partial_image = to_explain[0] * background_mask + mask * neutral_val
        partial_image = np.expand_dims(partial_image, axis=0).astype(np.float32)
        
        score = get_score(model, device, partial_image, trueImageInd)
        
        if abs(score - target_score) < epsilon:
            break
        elif score > target_score:
            low = mid
        else:
            high = mid
            
    return mid, score


def deletion_score_batch(
    to_explain: np.ndarray,
    trueImageInd: int,
    val: np.ndarray,
    model: torch.nn.Module,
    device: str,
    percentile_list: List[float],
    neutral_val: np.ndarray = 0,
    image_show: bool = False
) -> List[float]:
    """Compute deletion scores for a batch of percentile thresholds.
    
    For each percentile threshold, pixels with importance >= threshold are
    deleted and replaced with neutral value, then the model score is computed.
    
    Args:
        to_explain: Original image array with shape (B, H, W, C)
        trueImageInd: Target class index
        val: Importance values with shape (H, W)
        model: PyTorch model for inference
        device: Device to run inference on
        percentile_list: List of percentile thresholds (0-100)
        neutral_val: Value to replace deleted pixels with
        image_show: Whether to display images (for debugging)
        
    Returns:
        List of model scores for each percentile threshold
    """
    import matplotlib.pyplot as plt
    
    batch_partial_images = []
    
    for percentile in percentile_list:
        threshold = np.percentile(val, percentile)
        mask = val >= threshold
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=-1)
        background_mask = 1 - mask
        
        partial_image = to_explain[0] * background_mask + mask * neutral_val
        partial_image = partial_image.astype(np.float32)
        
        batch_partial_images.append(partial_image)
        
        if image_show:
            plt.imshow(partial_image)
            plt.show()
            
    return get_partial_score_batch(batch_partial_images, model, device, trueImageInd)


def get_auc_deletion(
    to_explain: np.ndarray,
    trueImageInd: int,
    val: np.ndarray,
    model: torch.nn.Module,
    device: str,
    x_values: Optional[np.ndarray] = None,
    neutral_value: np.ndarray = 0,
    image_show: bool = False
) -> float:
    """Compute area under the deletion curve.
    
    The deletion curve plots model confidence as increasingly important pixels
    are deleted. The area under this curve (AUC) measures the quality of the
    attribution: lower AUC means better attributions.
    
    Args:
        to_explain: Original image array with shape (B, H, W, C)
        trueImageInd: Target class index
        val: Importance values with shape (H, W)
        model: PyTorch model for inference
        device: Device to run inference on
        x_values: Percentile values to evaluate at. If None, uses 0-99.
        neutral_value: Value to replace deleted pixels with
        image_show: Whether to display images (for debugging)
        
    Returns:
        Area under the deletion curve
    """
    if x_values is None:
        x_values = np.arange(0, 100)
        
    deletion_score_list = deletion_score_batch(
        to_explain, 
        trueImageInd, 
        val, 
        model, 
        device, 
        x_values, 
        neutral_value, 
        image_show
    )
    
    y_values = np.array(deletion_score_list)
    area_under_curve = np.trapz(y_values, x_values)
    
    return area_under_curve

