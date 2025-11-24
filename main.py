"""Main script for generating saliency maps using Weighted Integrated Gradients."""

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms

from src.explainers import WGExplainer
from src.utils import (
    get_sample_image,
    normalize,
    get_neutral_background,
    plot_saliency_with_topk,
    load_model,
    get_available_models,
)
from src.utils.model_utils import get_true_id
from src.metrics import exact_find_d_alpha
from src.config import set_random_seed, DEFAULT_RANDOM_SEED


# Configuration
NUM_SAMPLES = 3
RANDOM_SEED = 6

# Set random seed for reproducibility
set_random_seed(RANDOM_SEED)


def explain_image(
    img_path: str,
    transform,
    model: torch.nn.Module,
    device: str,
    num_points: int = 500,
    img_alpha: float = 0.3,
    sali_alpha: float = 0.6
) -> tuple:
    """Generate weighted baseline explanation for an image.
    
    Args:
        img_path: Path to input image
        transform: Torchvision transform to apply
        model: PyTorch model for inference
        device: Device to run inference on
        num_points: Number of top attribution points to display
        img_alpha: Alpha transparency for original image
        sali_alpha: Alpha transparency for saliency map
        
    Returns:
        Tuple of (figure, axes) with the saliency visualization
    """
    # Load and prepare image
    transformed_image = get_sample_image(img_path, transform)
    to_explain = np.array(transformed_image.permute(1, 2, 0).unsqueeze(0))

    # Create multiple baselines
    white_baseline = np.ones(to_explain.shape)
    black_baseline = np.zeros(to_explain.shape)
    random_baseline = np.random.rand(*to_explain.shape)
    baseline = np.concatenate([white_baseline, black_baseline, random_baseline], axis=0)
    
    # Get model prediction and neutral background
    true_image_ind = get_true_id(to_explain, model, device)
    average_all_corners_broadcasted = get_neutral_background(to_explain[0])
    
    # Normalize baseline and create explainer
    normalized_baseline = normalize(baseline).to(device)
    explainer = WGExplainer(model.to(device), normalized_baseline, local_smoothing=0)

    # Compute SHAP values
    shap_values, indexes, baseline_samples, individual_grads = explainer.shap_values(
        normalize(to_explain).to(device),
        ranked_outputs=1,
        nsamples=NUM_SAMPLES,
        rseed=RANDOM_SEED
    )
    
    # Process SHAP values
    shap_values = [np.swapaxes(s, 0, -1) for s in shap_values]
    
    # Compute weights for each baseline
    weight_list = []
    for ind in range(len(individual_grads)):
        individual_grad = np.abs(individual_grads[ind])
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
    
    # Normalize weights (inverse relationship with deletion points)
    weight_list = np.array(weight_list)
    weight_list = (50176 / weight_list) / (50176 / weight_list).sum()
    weight_list = weight_list.reshape(-1, 1, 1, 1)
    
    # Compute weighted baseline attribution
    weighted_baseline_shap_val = np.sum(weight_list * individual_grads, axis=(0, 1))
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(4, 4))
    ax = plot_saliency_with_topk(
        ax,
        weighted_baseline_shap_val,
        to_explain.squeeze().copy(),
        "Weighted Baseline SHAP",
        k=num_points,
        img_alpha=img_alpha,
        sali_alpha=sali_alpha
    )
    
    return fig, ax


def main():
    """Main entry point for the script."""
    # Create argument parser
    parser = argparse.ArgumentParser(
        description='Generate saliency map for an image using Weighted Integrated Gradients'
    )
    parser.add_argument('--input', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, required=True, help='Output directory for saving results')
    parser.add_argument(
        '--model',
        type=str,
        default='vgg16',
        choices=get_available_models(),
        help=f'Model to use. Available: {", ".join(get_available_models())}'
    )
    parser.add_argument('--num-points', type=int, default=1000, help='Number of top points to display')
    parser.add_argument('--img-alpha', type=float, default=0.75, help='Alpha for original image')
    parser.add_argument('--sali-alpha', type=float, default=0.8, help='Alpha for saliency map')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Setup transform and model
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model, device=None)

    # Get output path
    img_basename = os.path.splitext(os.path.basename(args.input))[0]
    output_path = os.path.join(args.output, f"{img_basename}_{args.model}_explained.png")

    print(f"Processing image: {args.input}")
    print(f"Using model: {args.model}")

    # Generate and save the saliency map
    fig, ax = explain_image(
        args.input,
        transform,
        model,
        device,
        num_points=args.num_points,
        img_alpha=args.img_alpha,
        sali_alpha=args.sali_alpha
    )

    print(f"Saving result to: {output_path}")
    fig.savefig(output_path)
    print("Done!")


if __name__ == '__main__':
    main()
