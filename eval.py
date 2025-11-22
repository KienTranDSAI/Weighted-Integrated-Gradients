"""Evaluation script for Weighted Integrated Gradients on a dataset."""

import os
import random
import numpy as np
import torch
from torchvision import models, transforms

from src.explainers import IGExplainer
from src.utils import (
    get_sample_data,
    normalize,
    get_neutral_background,
)
from src.utils.model_utils import get_true_id
from src.metrics import exact_find_d_alpha, get_auc_deletion, filter_weights
from src.config import set_random_seed, DEFAULT_NUM_IMAGES, DEFAULT_THRESHOLD


# Configuration
RANDOM_SEED = 8
NUM_SAMPLES = 6
LOCAL_SMOOTHING = 0
NUM_IMAGES = DEFAULT_NUM_IMAGES
THRESHOLD = DEFAULT_THRESHOLD

# Set random seed for reproducibility
set_random_seed(RANDOM_SEED)


def evaluate(
    img_paths: list,
    mask_paths: list,
    transform,
    model: torch.nn.Module,
    device: str
) -> tuple:
    """Evaluate attribution methods on a dataset.
    
    This function computes and compares three attribution methods:
    1. Standard Integrated Gradients (SHAP)
    2. Weighted baseline attribution
    3. Filtered weighted baseline attribution
    
    Args:
        img_paths: List of image file paths
        mask_paths: List of mask file paths
        transform: Torchvision transform to apply
        model: PyTorch model for inference
        device: Device to run inference on
        
    Returns:
        Tuple of (shap_deletion_scores, weighted_deletion_scores, filtered_deletion_scores)
    """
    # Setup random seed and image indices
    total_images = len(img_paths)
    random.seed(RANDOM_SEED)
    image_indices = list(range(total_images))
    random.shuffle(image_indices)
    
    # Prepare raw image baseline (used by all images)
    raw_image_baseline = np.array(
        get_sample_data(0, img_paths, mask_paths, transform)[1]
        .permute(1, 2, 0)
        .unsqueeze(0)
    )
    
    # PHASE 1: Compute all IG explanations
    print("PHASE 1: Computing IG explanations...")
    shap_value_list = []
    grad_list = []
    
    for image_ind in range(NUM_IMAGES):
        random_ind = image_indices[image_ind]
        image_raw_name, transformed_image, transformed_mask = get_sample_data(
            random_ind, img_paths, mask_paths, transform
        )
        to_explain = np.array(transformed_image.permute(1, 2, 0).unsqueeze(0))
        
        # Create baselines
        white_baseline = np.ones(to_explain.shape)
        black_baseline = np.zeros(to_explain.shape)
        median_baseline = np.ones(to_explain.shape) * 0.5
        random_baseline = np.random.rand(*to_explain.shape)
        random_baseline1 = np.random.rand(*to_explain.shape)
        baseline = np.concatenate(
            [black_baseline, raw_image_baseline, random_baseline,
             random_baseline1, white_baseline, median_baseline],
            axis=0
        )
        
        true_image_ind = get_true_id(to_explain, model, device)
        average_all_corners_broadcasted = get_neutral_background(to_explain[0])
        normalized_baseline = normalize(baseline).to(device)
        explainer = IGExplainer(
            model.to(device),
            normalized_baseline,
            local_smoothing=LOCAL_SMOOTHING
        )
        
        # Get gradient from SHAP values
        shap_values, indexes, baseline_samples, individual_grads = explainer.shap_values(
            normalize(to_explain).to(device),
            ranked_outputs=1,
            nsamples=NUM_SAMPLES,
            rseed=RANDOM_SEED
        )
        shap_values = [np.swapaxes(s, 0, -1) for s in shap_values]
        raw_shap_value = np.sum(shap_values[0], axis=(0, -1))
        
        shap_value_list.append(raw_shap_value)
        grad_list.append(individual_grads)
    
    # PHASE 2: Evaluation
    print("\nPHASE 2: Evaluation...")
    weighted_baseline_value_list = []
    shap_deletion_score = []
    weighted_baseline_deletion_score = []
    filtered_weighted_baseline_deletion_score = []
    list_num_remove = []
    weights_array = []
    filtered_weights_array = []
    
    for image_ind in range(NUM_IMAGES):
        random_ind = image_indices[image_ind]
        print(f"Image path: {img_paths[random_ind]}")
        
        image_raw_name, transformed_image, transformed_mask = get_sample_data(
            random_ind, img_paths, mask_paths, transform
        )
        to_explain = np.array(transformed_image.permute(1, 2, 0).unsqueeze(0))
        true_image_ind = get_true_id(to_explain, model.to('cpu'))
        average_all_corners_broadcasted = get_neutral_background(to_explain[0])
        individual_grads = grad_list[image_ind]
        
        # Compute weights for each baseline
        weight_list = []
        for ind in range(len(individual_grads)):
            individual_grad = individual_grads[ind]
            individual_val = np.sum(individual_grad, axis=0)
            
            num_of_deleted_point, score = exact_find_d_alpha(
                model,
                device,
                to_explain,
                individual_val,
                trueImageInd=true_image_ind,
                target_ratio=0.5,
                neutral_val=average_all_corners_broadcasted,
                epsilon=0.005,
                max_iter=100
            )
            weight_list.append(num_of_deleted_point)
        
        # Normalize weights
        weight_list = np.array(weight_list)
        weight_list = (50176 / weight_list) / (50176 / weight_list).sum()
        
        print(f"Image id: {random_ind}")
        print(f"weight_list: {weight_list}")
        weights_array.append(weight_list)
        
        # Filter weights
        filtered_weight_list, num_remove = filter_weights(weight_list, THRESHOLD)
        list_num_remove.append(num_remove)
        
        print(f"filtered_weight_list: {filtered_weight_list}")
        filtered_weights_array.append(filtered_weight_list)
        
        weight_list = weight_list.reshape(-1, 1, 1, 1)
        filtered_weight_list = filtered_weight_list.reshape(-1, 1, 1, 1)
        
        # Compute weighted attributions
        weighted_baseline_shap_val = np.sum(weight_list * individual_grads, axis=(0, 1))
        filtered_weighted_baseline_shap_val = np.sum(
            filtered_weight_list * individual_grads, axis=(0, 1)
        )
        weighted_baseline_value_list.append(weighted_baseline_shap_val)
        
        # Compute deletion metrics
        raw_shap_value = shap_value_list[image_ind]
        print(f"shap sum: {raw_shap_value.sum()}, grad sum: {individual_grads[0].sum()}")
        
        shap_area_under_curve = get_auc_deletion(
            to_explain,
            true_image_ind,
            raw_shap_value,
            model,
            device,
            x_values=None,
            neutral_value=average_all_corners_broadcasted
        )
        
        weighted_baseline_area_under_curve = get_auc_deletion(
            to_explain,
            true_image_ind,
            weighted_baseline_shap_val,
            model,
            device,
            x_values=None,
            neutral_value=average_all_corners_broadcasted
        )
        
        filtered_weighted_baseline_area_under_curve = get_auc_deletion(
            to_explain,
            true_image_ind,
            filtered_weighted_baseline_shap_val,
            model,
            device,
            x_values=None,
            neutral_value=average_all_corners_broadcasted
        )
        
        shap_deletion_score.append(shap_area_under_curve)
        weighted_baseline_deletion_score.append(weighted_baseline_area_under_curve)
        filtered_weighted_baseline_deletion_score.append(
            filtered_weighted_baseline_area_under_curve
        )
        
        print("Area under the curve of raw shap:", shap_area_under_curve)
        print("Area under the curve of weighted baseline:", weighted_baseline_area_under_curve)
        print("Area under the curve of filtered weighted baseline:",
              filtered_weighted_baseline_area_under_curve)
        print()
    
    return shap_deletion_score, weighted_baseline_deletion_score, filtered_weighted_baseline_deletion_score


def main():
    """Main entry point for the evaluation script."""
    # Setup dataset paths
    # For local dataset
    dataset_dir = "./data"
    images_raw_names = [i for i in os.listdir(dataset_dir + "/Image")]
    img_paths = [dataset_dir + f"/Image/{name}" for name in images_raw_names]
    mask_paths = [dataset_dir + f"/Mask/{name}" for name in images_raw_names]
    
    # For Kaggle dataset (uncomment if needed)
    # dataset_dir = "/kaggle/input/old-data/Adaptive"
    # images_raw_names = [i for i in os.listdir(dataset_dir + "/Image")]
    # img_paths = [dataset_dir + f"/Image/{name}" for name in images_raw_names]
    # mask_paths = [dataset_dir + f"/Mask/{name}" for name in images_raw_names]
    
    # Setup transform and model
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Run evaluation
    shap_deletion_score, weighted_baseline_deletion_score, filtered_weighted_baseline_deletion_score = evaluate(
        img_paths, mask_paths, transform, model, device
    )
    
    # Print summary results
    print('-' * 50)
    print('SHAP DELETION: ', np.average(shap_deletion_score))
    print('WEIGHTED DELETION: ', np.average(weighted_baseline_deletion_score))
    print('FILTERED BASELINE: ', np.average(filtered_weighted_baseline_deletion_score))


if __name__ == "__main__":
    main()
