"""
predict.py
----------
Generates predictions on the Core50 test dataset using the trained MLP model.

Responsibilities:
    - Load saved normalization stats (mean + std)
    - Load test.csv and apply the same preprocessing used for training
    - Load the best saved model checkpoint (best_model.pth)
    - Run inference on the test samples
    - Create a submission CSV matching Kaggle's required format:
          id,label
          0,5
          1,12
          ...
    - Save the CSV to submissions/submission.csv

Usage:
    python src/predict.py
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from model import MLP
from data import Core50Dataset
from config import (
    TEST_PATH,
    MEAN_STD_PATH,
    BEST_MODEL_PATH,
    SUBMISSION_PATH,
    BATCH_SIZE
)


def generate_submission():
    """
    Loads the best model checkpoint, applies inference on the test dataset,
    and writes the Kaggle submission CSV to SUBMISSION_PATH.
    """

    # -------------------------
    # Load test.csv
    # -------------------------
    df = pd.read_csv(TEST_PATH)
    X_test = df.filter(regex="feature_").values  # shape: [49460, 500]

    # -------------------------
    # Load normalization stats from training
    # -------------------------
    stats = np.load(MEAN_STD_PATH)
    mean = stats["mean"]
    std = stats["std"]

    # Apply same normalization as training
    X_test_norm = (X_test - mean) / std

    # -------------------------
    # Wrap test features in Dataset + DataLoader
    # Dataset returns ONLY X (no labels)
    # -------------------------
    test_dataset = Core50Dataset(X_test_norm, y=None)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # -------------------------
    # Load model
    # -------------------------
    model = MLP()  # architecture must match training
    model.load_state_dict(torch.load(BEST_MODEL_PATH))  # load weights

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()  # evaluation mode

    all_preds = []  # store predicted class indices

    # -------------------------
    # Inference loop
    # -------------------------
    with torch.no_grad():
        for X_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)

            # Convert logits → predicted class index
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())

    # -------------------------
    # Build submission DataFrame
    # -------------------------
    submission_df = pd.DataFrame({
        "id": df["id"],        # keep original IDs
        "label": all_preds     # our predictions
    })

    # -------------------------
    # Save CSV to disk
    # -------------------------
    submission_df.to_csv(SUBMISSION_PATH, index=False)

    print(f"Submission file created at: {SUBMISSION_PATH}")
    print(f"Total predictions: {len(all_preds)}")


if __name__ == "__main__":
    generate_submission()
