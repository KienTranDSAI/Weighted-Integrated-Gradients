"""Overlap metrics for evaluating attribution quality using ground truth masks."""

import numpy as np
import copy
from typing import List

from .attribution_utils import create_mask_from_indices


def get_raw_important_point(important_val: np.ndarray, percentile: float) -> np.ndarray:
    """Get coordinates of important points above a percentile threshold.
    
    Args:
        important_val: 2D array of importance values
        percentile: Percentile threshold (0-100)
        
    Returns:
        Array of shape (N, 2) containing (row, col) coordinates of important points
    """
    threshold = np.percentile(important_val, percentile)
    indexes = np.where(important_val > threshold)
    
    second_dim = indexes[0]
    third_dim = indexes[1]
    datapoint = [[second_dim[i], third_dim[i]] for i in range(len(second_dim))]
    datapoint = np.array(datapoint)
    
    return datapoint


def create_shap_image(shap_value: np.ndarray, standard_threshold: float) -> np.ndarray:
    """Create a binary mask from SHAP values based on percentile threshold.
    
    Args:
        shap_value: 2D array of SHAP/attribution values
        standard_threshold: Percentile threshold (0-100)
        
    Returns:
        Binary mask with same shape as shap_value
    """
    important_point = get_raw_important_point(shap_value, standard_threshold)
    shap_image = np.zeros(shap_value.shape)
    
    for point in important_point:
        shap_image[point[0], point[1]] = 1
        
    return shap_image


def get_auc_overlapping(
    percentile_array: np.ndarray,
    val: np.ndarray,
    mask: np.ndarray,
    segment_points: int
) -> float:
    """Compute AUC of overlap between attribution and ground truth mask.
    
    This metric measures how well the most important pixels according to the
    attribution method overlap with the ground truth segmentation mask.
    Higher values indicate better overlap.
    
    Args:
        percentile_array: Array of percentile values to evaluate (e.g., 1-100)
        val: 2D array of importance/attribution values
        mask: Ground truth binary mask
        segment_points: Number of points in the ground truth mask
        
    Returns:
        Area under the overlap curve (AUC)
    """
    val = copy.deepcopy(val)
    mask = copy.deepcopy(mask)
    y_values = []
    
    for percentile in percentile_array:
        num_of_points = int(segment_points * percentile / 100)
        # Avoid division by zero
        if num_of_points == 0:
            y_values.append(0)
            continue
            
        image = create_mask_from_indices(val, num_of_points)
        overlap = np.sum(image * mask) / num_of_points
        y_values.append(overlap)
        
    return np.trapz(y_values, percentile_array)

