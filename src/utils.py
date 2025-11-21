"""
utils.py
--------
Contains helper functions used across the project.

Purpose:
    - Ensure full reproducibility with a set_seed() function
    - Provide a clean accuracy function for validation
"""

import torch
import numpy as np
import random


def set_seed(seed):
    """
    Sets seeds for Python, NumPy, and PyTorch
    to ensure reproducible results.

    Args:
        seed (int): The seed value defined in config.py
    """

    random.seed(seed)                  # Python RNG
    np.random.seed(seed)               # NumPy RNG
    torch.manual_seed(seed)            # CPU torch RNG
    torch.cuda.manual_seed_all(seed)   # GPU torch RNG (if available)


def accuracy_fn(logits, labels):
    """
    Computes classification accuracy.

    Args:
        logits (Tensor): Raw model outputs, shape [batch_size, num_classes]
        labels (Tensor): Ground truth class labels, shape [batch_size]

    Returns:
        float: Accuracy in range [0, 1]
    """

    # Convert logits to predicted class indices
    preds = torch.argmax(logits, dim=1)

    # Compare predictions to ground truth labels
    correct = (preds == labels).float().mean().item()

    return correct
