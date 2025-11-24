"""Configuration constants and settings for the Weighted Integrated Gradients library."""

import numpy as np
import torch

# ImageNet class names URL
IMAGENET_CLASS_INDEX_URL = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"

# Default image preprocessing parameters
DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]
DEFAULT_IMAGE_SIZE = (224, 224)


# Random seed for reproducibility
DEFAULT_RANDOM_SEED = 8

# Evaluation parameters
DEFAULT_NUM_SAMPLES = 6
DEFAULT_LOCAL_SMOOTHING = 0
DEFAULT_NUM_IMAGES = 41
DEFAULT_THRESHOLD = 0.25

# Deletion metric parameters
DEFAULT_TARGET_RATIO = 0.5
DEFAULT_EPSILON = 0.005
DEFAULT_MAX_ITER = 100

# Visualization parameters
DEFAULT_IMG_ALPHA = 0.3
DEFAULT_SALIENCY_ALPHA = 0.6
DEFAULT_TOP_K = 500


def set_random_seed(seed: int = DEFAULT_RANDOM_SEED) -> None:
    """Set random seed for reproducibility across numpy and PyTorch.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

