import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def train_mlp(
    X_train,
    y_train,
    hidden_dims=[128, 64],
    lr=1e-3,
    batch_size=64,
    epochs=1000,
    device="cpu",
):
    """
    Train a simple MLP on the provided training data.
    Args:
        X_train (np.ndarray): Training input data of shape (N, D).
        y_train (np.ndarray): Training output data of shape (N,).
        hidden_dims (list): List of hidden layer dimensions.
        lr (float): Learning rate for the optimizer.
        batch_size (int): Batch size for training.
        epochs (int): Number of training epochs.
        device (str): Device to use for training ('cpu' or 'cuda').
    """
    # Build DataLoader
    X = torch.tensor(X_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    ds = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    # Define simple MLP
    layers = []
    D = X_train.shape[1]
    prev = D
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers += [nn.Linear(prev, 1)]
    model = nn.Sequential(*layers).to(device)

    # Train
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()

    return model


def compute_feature_importance_from_data(
    X_train,
    y_train,
    hidden_dims=[128, 64],
    lr=1e-3,
    batch_size=64,
    epochs=100,
    device="cpu",
):
    """
    Compute gradient-based feature importance using a simple MLP.
    This function trains a model on the provided data and computes
    the feature importance based on the gradients of the output with
    respect to the inputs.
    Args:
        X_train (np.ndarray): Training input data of shape (N, D).
        y_train (np.ndarray): Training output data of shape (N,).
        hidden_dims (list): List of hidden layer dimensions.
        lr (float): Learning rate for the optimizer.
        batch_size (int): Batch size for training.
        epochs (int): Number of training epochs.
        device (str): Device to use for training ('cpu' or 'cuda').
    """
    # Train a simple MLP
    model = train_mlp(
        X_train,
        y_train,
        hidden_dims=hidden_dims,
        lr=lr,
        batch_size=batch_size,
        epochs=epochs,
        device=device,
    )

    # Compute gradients w.r.t. inputs
    model.eval()
    X_all = torch.tensor(X_train, dtype=torch.float32, device=device, requires_grad=True)# forward pass
    y_pred = model(X_all)
    # now backprop a uniform gradient of 1 over all outputs
    grad_outputs = torch.ones_like(y_pred)
    # Compute ∂y_pred / ∂X_all
    grads = torch.autograd.grad(
        outputs=y_pred,
        inputs=X_all,
        grad_outputs=grad_outputs,
        create_graph=False,
        retain_graph=False,
    )[0]  # shape [N, D]

    # Feature importance = mean absolute gradient across samples
    importances = grads.abs().mean(dim=0).cpu().numpy()  # shape (D,)

    return importances


def compute_feature_importance_from_model(
    model,
    X_train,
    device="cpu",
):
    """
    Compute feature importance from a trained model.
    Args:
        model (torch.nn.Module): Trained PyTorch model.
        X_train (np.ndarray): Training input data of shape (N, D).
        device (str): Device to use for computation ('cpu' or 'cuda').
    """
    # Ensure model is on correct device
    model.to(device)

    # Convert X_train to tensor
    X_all = torch.tensor(X_train, dtype=torch.float32, device=device, requires_grad=True)

    # Forward pass
    y_pred = model(X_all)

    # Compute gradients w.r.t. inputs
    grad_outputs = torch.ones_like(y_pred)
    grads = torch.autograd.grad(
        outputs=y_pred,
        inputs=X_all,
        grad_outputs=grad_outputs,
        create_graph=False,
        retain_graph=False,
    )[0]  # shape [N, D]

    # Feature importance = mean absolute gradient across samples
    importances = grads.abs().mean(dim=0).cpu().numpy()  # shape (D,)

    return importances