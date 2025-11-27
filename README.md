1. Project Overview

We implement a baseline MLP classifier trained on flattened image vectors.
The model is intentionally simple and designed to run quickly both locally and on Google Colab.

Goals of this baseline:

Provide a clean starting point for experimentation

Establish a consistent modular structure for the team

Achieve the required 50%+ validation accuracy

Current performance:

Best validation accuracy: 0.5213 (52.13%)

Kaggle submission accuracy: ~0.517

This meets the minimum competition requirements and serves as a baseline for future improvements.

2. Repository Structure
core50-kaggle/
│
├── data/                     # Not tracked (ignored)
│     ├── train.csv
│     ├── test.csv
│     └── sample_submission.csv
│
├── models/
│     ├── best_model.pth      # Saved model
│     └── mean_std.npz        # Normalization statistics
│
├── notebooks/
│     └── Comp_432_project.ipynb
│
├── src/
│     ├── config.py
│     ├── data.py
│     ├── model.py
│     ├── train.py
│     ├── predict.py
│     └── utils.py
│
├── submissions/
│     └── submission.csv
│
├── .gitignore
├── README.md
└── requirements.txt

3. File description
src/config.py

Defines project-wide configuration values including batch size, learning rate, epoch count, model dimensions, and file paths.

src/data.py

Handles loading of CSV files, feature extraction, standardization (mean and standard deviation), and the 80/20 train–validation split.
Returns PyTorch Dataset and DataLoader objects.

src/model.py

Defines the MLP architecture: Linear → ReLU → Dropout → Linear → Output.
A simple, fully connected network without convolutional layers.

src/train.py

Runs the full training loop for 25 epochs.
Tracks loss and validation accuracy, saves the best model, and writes normalization statistics to mean_std.npz.

src/predict.py

Loads the best saved model and normalization statistics, performs inference on the test set, and writes Kaggle-formatted predictions to submissions/submission.csv.

src/utils.py

Helper utilities for saving/loading model files, managing normalization statistics, and computing accuracy.

# Google Colab setup and instructions

the colab notebook can be found at:
notebooks/Comp_432_project.ipynb

this file does the following:

clones the repository

installs dependencies

uploads dataset files

runs training

generates a submission

downloads the output file

This notebook provides a full reproducible pipeline.

4. Running the Project Locally
1. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate

2. Install dependencies
pip install -r requirements.txt

3. Add CSV files

Place the dataset in:

data/train.csv
data/test.csv
data/sample_submission.csv

4. Train the model
python src/train.py

5. Generate a submission
python src/predict.py


Output is written to:

submissions/submission.csv

5. Running the Project in Google Colab

Use:

notebooks/Comp_432_project.ipynb


The notebook handles repository cloning, environment setup, file upload, training, prediction, and downloading the final submission file.

6. Current Results

Best validation accuracy: 0.5213

Kaggle submission score: ~0.517

The generated submission contains 49,460 predictions matching the expected competition format.