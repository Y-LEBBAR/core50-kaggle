"""
train.py
--------
Contains the full training loop for the MLP model.

Responsibilities:
    - Load train/validation DataLoaders from data.py
    - Initialize model, optimizer, loss function
    - Train for N epochs
    - Evaluate accuracy on validation set each epoch
    - Save the best model to disk

This script is the main entry point for training:

    python src/train.py
"""

import torch
import torch.nn as nn
import torch.optim as optim

from model import MLP
from data import load_train_val
from utils import set_seed, accuracy_fn
from config import (
    EPOCHS,
    LR,
    SEED,
    BEST_MODEL_PATH
)


def train():
    """
    Executes full training pipeline.

    - Loads data
    - Sets seeds
    - Builds model + optimizer
    - Trains for EPOCHS
    - Tracks best validation accuracy
    - Saves best model checkpoint
    """

    # Ensure reproducibility
    set_seed(SEED)

    # Load training and validation DataLoaders
    train_loader, val_loader = load_train_val()

    # Initialize model (500 → 256 → 50)
    model = MLP()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Loss function = CrossEntropy for classification
    criterion = nn.CrossEntropyLoss()

    # Optimizer = Adam (good default for MLPs)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0  # Track best validation score

    # -------------------------
    # Training loop
    # -------------------------
    for epoch in range(1, EPOCHS + 1):

        model.train()  # enable training mode
        running_loss = 0.0

        # Iterate over mini-batches
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()         # clear old gradients
            logits = model(X_batch)       # forward pass
            loss = criterion(logits, y_batch)  # compute loss
            loss.backward()               # backpropagate gradients
            optimizer.step()              # update parameters

            running_loss += loss.item()

        # -------------------------
        # Validation phase
        # -------------------------
        model.eval()  # disable dropout, batchnorm, etc.

        val_acc_total = 0.0
        val_batches = 0

        with torch.no_grad():  # disable gradients for speed
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                val_logits = model(X_val)
                acc = accuracy_fn(val_logits, y_val)
                val_acc_total += acc
                val_batches += 1

        val_acc = val_acc_total / val_batches

        print(f"Epoch [{epoch}/{EPOCHS}] | "
              f"Train Loss: {running_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")

        # Save model if validation improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"--> New best model saved (val_acc={best_val_acc:.4f})")

    print("\nTraining complete.")
    print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    train()
