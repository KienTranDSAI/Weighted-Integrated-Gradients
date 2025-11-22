"""Attribution utilities for processing and filtering importance scores."""

import numpy as np
from typing import Tuple, Optional


def get_sorted_indices(val: np.ndarray) -> np.ndarray:
    """Get indices that would sort a flattened array.
    
    Args:
        val: Input array to sort
        
    Returns:
        Indices that would sort the flattened array in ascending order
    """
    flattened_array = val.flatten()
    sorted_indices = np.argsort(flattened_array)
    return sorted_indices


def get_top_k(val: np.ndarray, k: int, sorted_indices: Optional[np.ndarray] = None) -> np.ndarray:
    """Find indices of top k largest values.
    
    Args:
        val: Input array to find top-k from
        k: Number of top values to find
        sorted_indices: Pre-computed sorted indices (optional)
        
    Returns:
        Indices of top k largest values in flattened array
    """
    if sorted_indices is None:
        sorted_indices = get_sorted_indices(val)
    top_k_indices = sorted_indices[-k:]
    return top_k_indices


def create_mask_from_indices(
    val: np.ndarray, 
    k: int, 
    sorted_indices: Optional[np.ndarray] = None
) -> np.ndarray:
    """Create a binary mask from the indices of the top k largest values.
    
    Args:
        val: Input array to create mask from
        k: Number of top values to mask
        sorted_indices: Pre-computed sorted indices (optional)
        
    Returns:
        Binary mask with same shape as val, with 1s at top-k positions
    """
    top_k_indices = get_top_k(val, k, sorted_indices)
    mask = np.zeros_like(val, dtype=int)
    top_k_positions = np.unravel_index(top_k_indices, val.shape)
    mask[top_k_positions] = 1
    return mask


def filter_weights(weights: np.ndarray, threshold: float) -> Tuple[np.ndarray, int]:
    """Filter and renormalize weights based on a threshold.
    
    Weights below threshold * mean are set to zero, unless the maximum weight
    is below the threshold, in which case only the maximum is kept.
    
    Args:
        weights: Array of weight values to filter
        threshold: Threshold multiplier for the mean weight
        
    Returns:
        Tuple of (filtered and normalized weights, number of weights removed)
    """
    num_remove = 0
    weights_array = np.array(weights)
    mean_weight = np.mean(weights_array)
    imax = np.argmax(weights_array)
    
    # If max weight is below threshold, keep only the max
    if weights[imax] < mean_weight * threshold:
        weights_array = np.zeros_like(weights_array)
        weights_array[imax] = weights[imax]
        num_remove = len(weights_array) - 1
    else:
        # Filter weights below threshold
        for i in range(len(weights_array)):
            if weights_array[i] < threshold * mean_weight:
                weights_array[i] = 0
                num_remove += 1

    # Renormalize
    weights_sum = np.sum(weights_array)
    if weights_sum > 0:
        weights_array = weights_array / weights_sum
    
    return weights_array, num_remove

