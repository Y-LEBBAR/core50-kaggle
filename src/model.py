"""
model.py
--------
Defines the baseline MLP (Multilayer Perceptron) architecture used for the
Core50 classification task.

This model is intentionally simple:
    Input (500 features) → Hidden Layer (256 units) → ReLU → Output (50 classes)

Reason:
    - Easy to train
    - Easy to read and understand
    - Modular: can be extended to deeper networks later
    - Rule-compliant: uses only basic PyTorch layers

This file contains ONLY the model definition.
All training logic is in train.py.
All data loading logic is in data.py.
"""

import torch.nn as nn


class MLP(nn.Module):
    """
    Basic 1-hidden-layer MLP for classification.

    Args:
        input_dim (int): Number of input features (default = 500)
        hidden_dim (int): Size of hidden layer (default = 256)
        num_classes (int): Number of output classes (default = 50)

    Forward:
        x -> logits of shape [batch_size, num_classes]
    """

    def __init__(self, input_dim=500, hidden_dim=256, num_classes=50):
        super().__init__()

        # Linear layer #1:
        # Takes the 500-dimensional input features and maps them to 256 hidden units.
        self.hidden = nn.Linear(input_dim, hidden_dim)

        # ReLU activation:
        # Introduces non-linearity so the network can learn complex relationships.
        self.relu = nn.ReLU()

        # Linear layer #2:
        # Maps hidden_dim units → num_classes logits (no softmax needed; CrossEntropyLoss handles that)
        self.output_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (Tensor): Input tensor of shape [batch_size, input_dim]

        Returns:
            Tensor: Logits of shape [batch_size, num_classes]
        """

        # First linear transformation
        x = self.hidden(x)

        # Activation function
        x = self.relu(x)

        # Output layer: raw class logits
        x = self.output_layer(x)

        return x
