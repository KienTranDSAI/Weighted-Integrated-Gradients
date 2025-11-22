"""Visualization utilities for saliency maps and explanations."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def plot_saliency_with_topk(
    ax: plt.Axes,
    saliency: np.ndarray, 
    original_image: np.ndarray, 
    title: str, 
    k: int = 500, 
    cmap: str = 'hot', 
    img_alpha: float = 0.3, 
    sali_alpha: float = 0.6
) -> plt.Axes:
    """Plot saliency map showing only top-k values overlayed on the original image.
    
    Args:
        ax: Matplotlib axes to plot on
        saliency: Array of saliency/importance values to visualize
        original_image: The original image to overlay
        title: Title for the saliency plot
        k: Number of top values to display
        cmap: Colormap to use for saliency visualization
        img_alpha: Alpha value for the original image overlay (lower = more transparent)
        sali_alpha: Alpha value for the saliency map overlay
        
    Returns:
        The matplotlib axes with the plot
    """
    # Normalize saliency values
    saliency = saliency.copy()
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())

    # Create masked version showing only top-k values
    topk_saliency = np.zeros_like(saliency)
    
    # Get sorted indices
    flat_saliency = saliency.flatten()
    sorted_indices = np.argsort(flat_saliency)[::-1]  # Sort in descending order
    
    # Get top-k indices
    top_k_indices = sorted_indices[:k]
    
    # Create masked version with only top-k values
    flattened_topk = np.zeros_like(flat_saliency)
    flattened_topk[top_k_indices] = flat_saliency[top_k_indices]
    topk_saliency = flattened_topk.reshape(saliency.shape)
    
    # Set background color
    ax.set_facecolor('white')
    
    # First show the original image with low alpha
    ax.imshow(original_image, cmap='hot', alpha=img_alpha)
    
    # Then overlay the saliency map
    im = ax.imshow(topk_saliency, cmap=cmap, alpha=sali_alpha)
    
    # Set title and remove ticks
    ax.set_title(title, color='white', fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add white border
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
        spine.set_linewidth(2)

    plt.tight_layout()
    return ax


def create_shap_image(shap_value: np.ndarray, standard_threshold: float) -> np.ndarray:
    """Create binary mask from SHAP values using percentile threshold.
    
    Args:
        shap_value: SHAP attribution values with shape (H, W)
        standard_threshold: Percentile threshold (0-100) for selecting important points
        
    Returns:
        Binary mask with shape (H, W) where 1 indicates important pixels
    """
    threshold = np.percentile(shap_value, standard_threshold)
    indexes = np.where(shap_value > threshold)
    
    shap_image = np.zeros(shap_value.shape)
    second_dim = indexes[0]
    third_dim = indexes[1]
    
    for i in range(len(second_dim)):
        shap_image[second_dim[i], third_dim[i]] = 1
        
    return shap_image

