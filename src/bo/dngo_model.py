"""
Deep Networks for Global Optimization. This module performs bayesian linear regression with basis function
extracted from a feed forward neural network.
Reference: [1] J. Snoek, O. Rippel, K. Swersky, R. Kiros, N. Satish,
            N. Sundaram, M.~M.~A. Patwary, Prabhat, R.~P. Adams
            Scalable Bayesian Optimization Using Deep Neural Networks
            Proc. of ICML'15
This implementation is a purely PyTorch based rewrite of the following code sources:
- https://github.com/janschwedhelm/master-thesis/blob/main/src/dngo/dngo.py
- https://github.com/janschwedhelm/master-thesis/blob/main/src/dngo/base_model.py
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.bo.bayesian_linear_regression import BayesianLinearRegression, Prior
from src.utils import zero_mean_unit_var_normalization, zero_mean_unit_var_denormalization


class Net(nn.Module):
    """
    Feed forward neural network with 3 hidden layers and tanh activation function.
    """
    def __init__(self, n_inputs, n_units=[50, 50, 50]):
        """
        Initializes the neural network.
        Args:
            n_inputs (int): Number of input features
            n_units (list): Number of units in each hidden layer
        """
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_inputs, n_units[0])
        self.fc2 = nn.Linear(n_units[0], n_units[1])
        self.fc3 = nn.Linear(n_units[1], n_units[2])
        self.out = nn.Linear(n_units[2], 1)

    def forward(self, x):
        """
        Forward pass through the network.
        Args:
            x (torch.Tensor): Input tensor
        Returns:
            torch.Tensor: Output tensor
        """
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return self.out(x)

    def basis_funcs(self, x):
        """
        Computes the basis functions for the input data.
        Args:
            x (torch.Tensor): Input tensor
        Returns:
            torch.Tensor: Output tensor
        """
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x
    

class DNGO:
    """
    Deep Networks for Global Optimization (DNGO) model.
    This class implements a Bayesian optimization model using deep neural networks
    to extract features and Bayesian linear regression for prediction.
    """
    def __init__(self, batch_size=64, num_epochs=500, learning_rate=0.01,
                 n_units=[200, 50, 50], alpha=1.0, beta=1000, prior=None,
                 normalize_input=True, normalize_output=True, device=None):
        """
        Initializes the DNGO model.
        Args:
            batch_size (int): Batch size for training the neural network
            num_epochs (int): Number of epochs for training
            learning_rate (float): Learning rate for Adam optimizer
            n_units (list): Number of units in each hidden layer
            alpha (float): Hyperparameter of the Bayesian linear regression
            beta (float): Hyperparameter of the Bayesian linear regression
            prior (Prior object): Prior for alpha and beta. If None, the default prior is used
            normalize_input (bool): If True, normalize input data to zero mean and unit variance
            normalize_output (bool): If True, normalize output data to zero mean and unit variance
            device (str): Device to run the model on ('cpu' or 'cuda'). If None, uses the default device.
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        # Store high-level parameters
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.n_units = n_units

        self.normalize_input = normalize_input
        self.normalize_output = normalize_output

        # Prior over (log alpha, log beta)
        self.prior = prior if prior else Prior(device=self.device)

        # Initialize hyperparameters for BLR
        self.alpha0 = alpha
        self.beta0 = beta

        # Will be filled during training
        self.X_mean = self.X_std = None
        self.y_mean = self.y_std = None
        self.network = None
        self.blr = None
        self.hypers = None

    def train(self, X, y, do_optimize=True):
        """
        Trains the DNGO model on the provided data.
        Args:
            X (torch.Tensor): Input data points with shape (N, D).
            y (torch.Tensor): Corresponding target values with shape (N,).
            do_optimize (bool): If True, optimize hyperparameters, otherwise sample them.
        """
        # Move data to device, store raw data for later use
        X = X.to(self.device, dtype=torch.float32)
        y = y.to(self.device, dtype=torch.float32).view(-1)
        self._X_raw = X.clone()
        self._y_raw = y.clone()

        # Normalize if needed
        if self.normalize_input:
            X, self.X_mean, self.X_std = zero_mean_unit_var_normalization(X)
        if self.normalize_output:
            y, self.y_mean, self.y_std = zero_mean_unit_var_normalization(y)

        # Create the neural network
        self.network = Net(n_inputs=X.shape[1], n_units=self.n_units).to(self.device)

        optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=10, verbose=True)

        dataset = torch.utils.data.TensorDataset(X, y.unsqueeze(1))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Train network
        self.network.train()
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            for xb, yb in dataloader:
                optimizer.zero_grad()
                pred = self.network(xb)
                loss = F.mse_loss(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)

            scheduler.step(epoch_loss / len(dataset))

            if (epoch + 1) % 10 == 0:
                logging.info(f"Epoch [{epoch + 1}/{self.num_epochs}], Loss: {epoch_loss / len(dataset):.4f}")

            if optimizer.param_groups[0]['lr'] < 1e-6:
                logging.info("Learning rate too low, stopping training.")
                break

        # Build design matrix (features)
        self.network.eval()
        with torch.no_grad():
            Theta = self.network.basis_funcs(X).detach()

        # Train Bayesian Linear Regression model
        self.blr = BayesianLinearRegression(
            alpha=self.alpha0,
            beta=self.beta0,
            basis_func=None,  # Will use Theta directly
            prior=self.prior,
            device=self.device
        )
        self.blr.train(Theta, y, do_optimize=do_optimize)
        self.hypers = [self.blr.alpha.item(), self.blr.beta.item()]

        logging.info(f"Finished DNGO training. Hypers: alpha={self.hypers[0]}, beta={self.hypers[1]}")

    def negative_mll(self, theta):
        """
        Negative marginal log likelihood for the Bayesian linear regression on top of the frozen neural network features.
        Args:
            theta (torch.Tensor): Hyperparameters of the Bayesian linear regression (log alpha, log beta).
        Returns:
            torch.Tensor: The negative marginal log likelihood.
        """
        return -self.blr.marginal_log_likelihood(theta)

    def predict(self, X_test):
        """
        Make predictions using the trained DNGO model.
        Args:
            X_test (torch.Tensor): Input data points with shape (N, D).
        Returns:
            torch.Tensor: Predicted mean of shape (N,).
            torch.Tensor: Predicted variance of shape (N,).
        """
        # Move to device and normalize if needed
        X_test = X_test.to(self.device, dtype=torch.float32)
        if self.normalize_input:
            X_test = zero_mean_unit_var_normalization(X_test, self.X_mean, self.X_std)

        # Forward pass through the network to get basis functions
        self.network.eval()
        Phi_test = self.network.basis_funcs(X_test)

        # BLR prediction
        mu, var = self.blr.predict(Phi_test)

        # Denormalize outputs if required
        if self.normalize_output:
            mu = zero_mean_unit_var_denormalization(mu, self.y_mean, self.y_std)
            var *= self.y_std ** 2

        return mu, var
    
    def get_incumbent(self):
        """
        Get the best observed point and its function value.
        Returns:
            torch.Tensor: The best observed point.
            float: The function value at the best observed point.
        """
        # Get inputs and targets from the training data
        X_tr = self._X_raw
        y_tr = self._y_raw

        # Index of the best (minimum) target value
        best_idx = torch.argmin(y_tr)

        x_best = X_tr[best_idx]
        y_best = y_tr[best_idx].item()
        
        return x_best, y_best