"""
config.py
---------
Stores all global configuration values for the Core50 Kaggle project.

Purpose:
    - Keep all constants in a single place
    - Make hyperparameter tuning simple
    - Allow teammates to modify settings without touching code logic
    - Ensure consistent paths across training/inference scripts
"""

# ----------------------------
# Training Hyperparameters
# ----------------------------

EPOCHS = 25              # Number of full passes through the training dataset
LR = 1e-3                # Learning rate for Adam optimizer
BATCH_SIZE = 512         # Samples per mini-batch
SEED = 42                # Reproducibility

# ----------------------------
# Dataset Info
# ----------------------------

INPUT_DIM = 500          # Each sample has 500 features
NUM_CLASSES = 50         # 50 object classes to predict

# ----------------------------
# File Paths
# ----------------------------

TRAIN_PATH = "data/train.csv"          # Local training file
TEST_PATH = "data/test.csv"            # Local test file

MEAN_STD_PATH = "models/mean_std.npz"  # Where we store normalization parameters
BEST_MODEL_PATH = "models/best_model.pth"  # Saved PyTorch model checkpoint
SUBMISSION_PATH = "submissions/submission.csv"  # Submission output file
