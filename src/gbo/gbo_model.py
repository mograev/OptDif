"""
Simple feedforward neural network model used for Gradient-Based Optimization (GBO).
This model is designed to be used with PyTorch and is built using the nn.Module class.
The model consists of multiple linear layers with ReLU activations in between.
"""
import torch
import torch.nn as nn

class GBOModel(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim=1,
    ):
        """
        Initialize the GBOModel.
        Args:
            input_dim (int): The number of input features.
            hidden_dims (list): A list of integers representing the number of units in each hidden layer.
            output_dim (int): The number of output features. Default is 1.
        """
        super().__init__()

        layers = []
        dims = [input_dim] + hidden_dims

        # Hidden layers
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
        # Final layer
        layers.append(nn.Linear(dims[-1], output_dim))
        layers.append(nn.Sigmoid())

        self.register_buffer("factors", torch.tensor([5.0] * output_dim))

        # Create the model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the model.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor after passing through the model.
        """
        return self.factors * self.model(x)