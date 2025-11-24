"""Evaluation script for Weighted Integrated Gradients on a dataset."""

import argparse
import os
import random
import numpy as np
import torch
from torchvision import transforms

from src.explainers import IGExplainer
from src.utils import (
    get_sample_data,
    normalize,
    get_neutral_background,
    load_model,
    get_available_models,
)
from src.utils.model_utils import get_true_id
from src.metrics import exact_find_d_alpha, get_auc_deletion, filter_weights
from src.config import set_random_seed, DEFAULT_NUM_IMAGES, DEFAULT_THRESHOLD, DEFAULT_IMAGE_SIZE

# Configuration
RANDOM_SEED = 8
NUM_SAMPLES = 6
LOCAL_SMOOTHING = 0
DEFAULT_NUM_IMAGES_EVAL = DEFAULT_NUM_IMAGES
THRESHOLD = DEFAULT_THRESHOLD
IMAGE_SIZE = DEFAULT_IMAGE_SIZE

# Set random seed for reproducibility
set_random_seed(RANDOM_SEED)

def evaluate(
    img_paths: list,
    mask_paths: list,
    transform,
    model: torch.nn.Module,
    device: str,
    num_images: int
) -> tuple:
    """Evaluate attribution methods on a dataset.
    
    This function computes and compares three attribution methods:
    1. Expected Gradients (EG)
    2. Weighted baseline attribution (WG)
    3. Filtered weighted baseline attribution (FWG)
    
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
    
    # Prepare a real image as baseline (used by all images)
    raw_image_baseline = np.array(
        get_sample_data(0, img_paths, mask_paths, transform)[1]
        .permute(1, 2, 0)
        .unsqueeze(0)
    )
    
    # PHASE 1: Compute intermediate IG explanations
    shap_value_list = []
    grad_list = []
    
    for image_ind in range(num_images):
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
    
    # PHASE 2: Weighted Baseline Evaluation
    weighted_baseline_value_list = []
    shap_deletion_score = []
    weighted_baseline_deletion_score = []
    filtered_weighted_baseline_deletion_score = []
    list_num_remove = []
    weights_array = []
    filtered_weights_array = []
    
    for image_ind in range(num_images):
        random_ind = image_indices[image_ind]
        # print(f"Image path: {img_paths[random_ind]}")
        image_raw_name, transformed_image, transformed_mask = get_sample_data(
            random_ind, img_paths, mask_paths, transform
        )
        to_explain = np.array(transformed_image.permute(1, 2, 0).unsqueeze(0))
        true_image_ind = get_true_id(to_explain, model, 'cpu')
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
        normalization_term = IMAGE_SIZE[0] * IMAGE_SIZE[1]
        weight_list = np.array(weight_list)
        weight_list = (normalization_term / weight_list) / (normalization_term / weight_list).sum()
        
        weights_array.append(weight_list)
        
        # Filter weights
        filtered_weight_list, num_remove = filter_weights(weight_list, THRESHOLD)
        list_num_remove.append(num_remove)
        
        filtered_weights_array.append(filtered_weight_list)
        
        weight_list = weight_list.reshape(-1, 1, 1, 1)
        filtered_weight_list = filtered_weight_list.reshape(-1, 1, 1, 1)
        
        # Compute weighted attributions
        weighted_baseline_shap_val = np.sum(weight_list * individual_grads, axis=(0, 1))
        filtered_weighted_baseline_shap_val = np.sum(
            filtered_weight_list * individual_grads, axis=(0, 1)
        )
        weighted_baseline_value_list.append(weighted_baseline_shap_val)
        
        # PHASE 3: Compute deletion metrics
        raw_shap_value = shap_value_list[image_ind]
        
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
        # print("Area under the curve of raw shap:", shap_area_under_curve)
        # print("Area under the curve of weighted baseline:", weighted_baseline_area_under_curve)
        # print("Area under the curve of filtered weighted baseline:", filtered_weighted_baseline_area_under_curve)
        # print("-" * 50)
        
    return shap_deletion_score, weighted_baseline_deletion_score, filtered_weighted_baseline_deletion_score


def main():
    """Main entry point for the evaluation script."""
    # Create argument parser
    parser = argparse.ArgumentParser(
        description='Evaluate Weighted Integrated Gradients on a dataset'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='vgg16',
        choices=get_available_models(),
        help=f'Model to use for evaluation. Available: {", ".join(get_available_models())}'
    )
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default='./data',
        help='Path to dataset directory containing Image/ and Mask/ folders'
    )
    parser.add_argument(
        '--num-images',
        type=int,
        default=DEFAULT_NUM_IMAGES_EVAL,
        help=f'Number of images to evaluate (default: {DEFAULT_NUM_IMAGES_EVAL})'
    )
    args = parser.parse_args()
    
    # Setup dataset paths
    dataset_dir = args.dataset_dir
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory '{dataset_dir}' does not exist.")
        print("Please specify a valid dataset directory with --dataset-dir")
        return
    
    images_raw_names = [i for i in os.listdir(dataset_dir + "/Image")]
    img_paths = [dataset_dir + f"/Image/{name}" for name in images_raw_names]
    mask_paths = [dataset_dir + f"/Mask/{name}" for name in images_raw_names]
    
    # Determine number of images to evaluate
    num_images_to_eval = min(args.num_images, len(img_paths))
    
    print(f"Dataset directory: {dataset_dir}")
    print(f"Found {len(img_paths)} images")
    print(f"Evaluating on {num_images_to_eval} images")
    print(f"Model: {args.model}")
    print("-" * 50)
    
    # Setup transform and model
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor()
    ])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model, device=None)
    
    # Run evaluation
    shap_deletion_score, weighted_baseline_deletion_score, filtered_weighted_baseline_deletion_score = evaluate(
        img_paths, mask_paths, transform, model, device, num_images_to_eval
    )
    
    # Print summary results
    print('\n' + '=' * 50)
    print('FINAL RESULTS')
    print('=' * 50)
    print(f'Model: {args.model.upper()}')
    print('-' * 50)
    print(f'Mean AUC of Deletion - EG :           {np.average(shap_deletion_score):.4f}')
    print(f'Mean AUC of Deletion - WG:            {np.average(weighted_baseline_deletion_score):.4f}')
    print(f'Mean AUC of Deletion - Filtered WG:   {np.average(filtered_weighted_baseline_deletion_score):.4f}')
    print('=' * 50)


if __name__ == "__main__":
    main()
