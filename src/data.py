"""
data.py
-------
Handles all data-related operations for the Core50 Kaggle project.

Responsibilities:
    - Load the training CSV file
    - Extract feature matrix (X) and labels (y)
    - Compute mean and standard deviation for normalization
    - Save normalization stats to disk for later use in prediction
    - Create an 80/20 train-validation split
    - Return PyTorch Datasets and DataLoaders

This file contains NO model or training logic.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from config import (
    TRAIN_PATH,
    MEAN_STD_PATH,
    BATCH_SIZE,
    SEED
)


class Core50Dataset(Dataset):
    """
    PyTorch Dataset for Core50 pre-extracted features.

    Args:
        X (ndarray): Normalized feature matrix of shape [N, 500]
        y (ndarray or None): Class labels of shape [N], or None for test data

    __getitem__ returns either:
        (X[i], y[i])   for training/validation
        X[i]           for test inference
    """

    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is None:  # used for test set
            return self.X[idx]
        return self.X[idx], self.y[idx]


def load_train_val():
    """
    Loads training data, normalizes it, and creates an 80/20 train-val split.

    Returns:
        train_loader (DataLoader): DataLoader for training set
        val_loader   (DataLoader): DataLoader for validation set
    """

    # -------------------------
    # Load CSV file
    # -------------------------
    df = pd.read_csv(TRAIN_PATH)

    # Extract all feature columns: feature_0, feature_1, ..., feature_499
    X = df.filter(regex="feature_").values
    y = df["label"].values  # shape: [num_samples]

    # -------------------------
    # Compute normalization stats
    # -------------------------
    mean = X.mean(axis=0)              # vector of shape [500]
    std = X.std(axis=0) + 1e-8         # avoid division-by-zero

    # Save stats to disk (used later by predict.py)
    np.savez(MEAN_STD_PATH, mean=mean, std=std)

    # -------------------------
    # Normalize features
    # -------------------------
    X_norm = (X - mean) / std

    # -------------------------
    # Train/Validation split (80/20)
    # Stratify ensures class distribution consistency
    # -------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X_norm,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y
    )

    # -------------------------
    # Create PyTorch datasets
    # -------------------------
    train_dataset = Core50Dataset(X_train, y_train)
    val_dataset = Core50Dataset(X_val, y_val)

    # -------------------------
    # Wrap in DataLoaders
    # -------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_loader, val_loader
