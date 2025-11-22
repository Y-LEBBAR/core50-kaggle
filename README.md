COMP 432 – CORE50 Kaggle Project

Simple MLP Baseline Classifier for Object Recognition

This repository contains the full codebase for our COMP 432 Kaggle competition submission.
The goal is to build a baseline model that classifies images in the CORE50 dataset using a simple Multilayer Perceptron (MLP).
All code is modular, fully commented, and designed to be improved later with deeper models.

## 1. Project Overview

We implement a baseline MLP classifier trained on flattened image vectors.
This model is intentionally simple (no convolution layers yet).
Its purpose is to:

establish a strong, clean baseline (~52% validation accuracy)

create a modular codebase for teammates

allow fast experimentation

run both locally and on Google Colab

This baseline already achieves:

Best validation accuracy: 0.5213 (52.13%) (0.517 on kaggle submission)

which surpasses the 50% threshold required for the competition baseline.

## 2. Repository Structure
core50-kaggle/
│
├── data/                     # NOT pushed to GitHub (.gitignore)
│     ├── train.csv
│     ├── test.csv
│     └── sample_submission.csv
│
├── models/
│     ├── best_model.pth      # Saved best PyTorch model
│     └── mean_std.npz        # Normalization statistics
│
├── notebooks/
│     └── Comp_432_project.ipynb  # Google Colab training notebook
│
├── src/
│     ├── config.py           # Global configuration
│     ├── data.py             # Dataset loading + preprocessing
│     ├── model.py            # MLP model definition
│     ├── train.py            # Training loop
│     ├── predict.py          # Inference + submission generation
│     └── utils.py            # Helpers (normalization, saving model, etc.)
│
├── submissions/
│     └── submission.csv      # Kaggle submission file
│
├── .gitignore
├── README.md
└── requirements.txt

## 3. Explanation of Each File
### 📁 src/config.py

Central configuration file. Defines:

batch size

learning rate

number of epochs

model size

file paths

This makes it easy to change settings across the entire project.

### 📁 src/data.py

Handles:

loading train.csv and test.csv

extracting input features and labels

applying standardization (mean/std)

splitting dataset into 80/20 train/validation

Output: PyTorch Dataset and DataLoader objects.

### 📁 src/model.py

Defines the MLP classifier:

Input → Linear → ReLU → Dropout → Linear → Output

Fully connected network

No convolution layers (simple baseline)

The model is intentionally small and fast.

### 📁 src/train.py

Main training loop:

loads data

trains model for 25 epochs

tracks training loss & validation accuracy

saves the best model automatically as models/best_model.pth

saves normalization stats (mean_std.npz)

Final validation result achieved:

Val Accuracy: 0.5213

### 📁 src/predict.py

Used to generate submission.csv:

loads the saved best model

loads test.csv

normalizes using training mean/std

outputs predictions in the exact Kaggle format

saves them to submissions/submission.csv

### 📁 src/utils.py

Utility functions including:

saving/loading model

saving/loading normalization stats

accuracy computation

small helper wrappers

### 📁 notebooks/Comp_432_project.ipynb

A full Google Colab notebook that walks through:

cloning repo

installing dependencies

uploading train/test/sample CSVs

running training (train.py)

running prediction (predict.py)

downloading submission.csv

uploading it to GitHub manually

This notebook is clean, ready for teammates, and contains all steps.

## 4. Running the Project Locally
### 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate

### 2. Install dependencies
pip install -r requirements.txt

### 3. Add CSV files

Place your dataset in:

data/train.csv
data/test.csv
data/sample_submission.csv

### 4. Train the model
python src/train.py

### 5. Generate submission
python src/predict.py


Output will be saved to:

submissions/submission.csv

## 5. Running With Google Colab

Use the notebook:

notebooks/Comp_432_project.ipynb


It performs:

repo cloning

environment setup

file uploads

training

prediction

downloading submission

Colab is recommended for faster CPU/GPU training.

## 6. Current Results

Our simple MLP baseline achieved:

Best validation accuracy: 0.5213

This satisfies the competition requirement and provides a strong benchmark.

Submission file contains:

49,460 predictions


Matching the exact expected format.

## 7. Next Steps: Improving Accuracy
🔥 1. Switch from MLP → Convolutional Neural Network (CNN)

MLP ignores spatial structure.
Even a small ConvNet (2 conv layers) will likely reach 70–80% accuracy.

🔥 2. Add data augmentation

Random:

flips

brightness shifts

cropping

rotation

This reduces overfitting and increases validation accuracy.

🔥 3. Use a deeper MLP

Add:

more layers

batch normalization

increased hidden units

🔥 4. Early stopping + learning rate scheduler

Improves stability.

🔥 5. Feature scaling using PCA

Dimensionality reduction before MLP may help.

🔥 6. Replace MLP with Logistic Regression / SVM baseline

For comparison.

🔥 7. Use PyTorch Lightning for cleaner training code
## 8. Team Workflow Recommendations

Use GitHub for collaboration

Keep .gitignore strict (no data pushed)

Use Colab for training

All team members should run the notebook to reproduce results

Every new model should be stored as a separate script inside src/models/

## 9. Credits

Team members: Yannis Lebbar + collaborators
Course: COMP 432 – Machine Learning
Dataset: CORE50 (Kaggle Competition)