"""
Train a GBO model on a subset of dimensions, that have a large feature importance.
"""

import logging
import argparse
import pickle
import time

from sklearn.decomposition import PCA
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.gbo.gbo_model import GBOModel
from src.utils import zero_mean_unit_var_normalization

# Arguments
parser = argparse.ArgumentParser()
gbo_train_group = parser.add_argument_group("GBO training")
gbo_train_group.add_argument("--seed", type=int, required=True)
gbo_train_group.add_argument("--logfile", type=str, help="file to log to", default="bo_train.log")
gbo_train_group.add_argument("--data_file", type=str, help="file to load data from", required=True)
gbo_train_group.add_argument("--save_file", type=str, required=True, help="path to save results to")
gbo_train_group.add_argument("--device", type=str, default="cpu", help="device to use for training (cpu or cuda)")
gbo_train_group.add_argument("--normalize_input", action="store_true", help="normalize input data")
gbo_train_group.add_argument("--normalize_output", action="store_true", help="normalize output data")
gbo_train_group.add_argument("--n_opt_dims", type=int, default=512, help="number of dimensions to optimize over")



def compute_feature_importance(
    X_train,
    y_train,
    hidden_dims=[128, 64],
    lr=1e-3,
    batch_size=64,
    epochs=100,
    device="cpu",
):
    """
    Compute feature importance using a simple MLP.
    Args:
        X_train (np.ndarray): Training input data of shape (N, D).
        y_train (np.ndarray): Training output data of shape (N,).
        hidden_dims (list): List of hidden layer dimensions.
        lr (float): Learning rate for the optimizer.
        batch_size (int): Batch size for training.
        epochs (int): Number of training epochs.
        device (str): Device to use for training ('cpu' or 'cuda').
    """
    # 1) Build DataLoader
    X = torch.tensor(X_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    ds = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    # 2) Define simple MLP
    layers = []
    D = X_train.shape[1]
    prev = D
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers += [nn.Linear(prev, 1)]
    model = nn.Sequential(*layers).to(device)

    # 3) Train
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

    # 4) Compute gradients w.r.t. inputs
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

    # 5) Feature importance = mean absolute gradient across samples
    importances = grads.abs().mean(dim=0).cpu().numpy()  # shape (D,)

    return importances


def train_gbo(
    logfile,
    data_file,
    save_file,
    device="cpu",
    normalize_input=True,
    normalize_output=True,
    n_opt_dims=512,
):
    """
    Train the GBO model.
    Args:
        logfile (str): Path to the log file.
        data_file (str): Path to the data file.
        save_file (str): Path to save the trained model.
        device (str): Device to use for training (cpu or cuda).
        normalize_input (bool): Whether to normalize the input data.
        normalize_output (bool): Whether to normalize the output data.
        n_opt_dims (int): Number of dimensions to optimize over.
    """

    # Set up logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG) # INFO
    logger.addHandler(logging.FileHandler(logfile))

    # Load the data
    with np.load(data_file, allow_pickle=True) as npz:
        X_train = npz['X_train']
        y_train = npz['y_train']

    # Reshape the data
    X_train = X_train.reshape(X_train.shape[0], -1)
    y_train = y_train.reshape(y_train.shape[0])
    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"y_train shape: {y_train.shape}")

    # Data statistics
    logger.debug(f"X_train stats: mean {X_train.mean()}, std {X_train.std()}, min {X_train.min()}, max {X_train.max()}")
    logger.debug(f"y_train stats: mean {y_train.mean()}, std {y_train.std()}, min {y_train.min()}, max {y_train.max()}")

    # Normalize inputs
    if normalize_input:
        logger.info("Normalizing input data")
        X_train, X_mean, X_std = zero_mean_unit_var_normalization(X_train)
        logger.debug(f"X_train stats after normalization: mean {X_train.mean()}, std {X_train.std()}, min {X_train.min()}, max {X_train.max()}")

    # Normalize outputs
    if normalize_output:
        logger.info("Normalizing output data")
        y_train, y_mean, y_std = zero_mean_unit_var_normalization(y_train)
        logger.debug(f"y_train stats after normalization: mean {y_train.mean()}, std {y_train.std()}, min {y_train.min()}, max {y_train.max()}")

    # Get feature importance
    logger.info("Computing feature importance")
    feature_importance = compute_feature_importance(X_train, y_train, device=device)
    logger.info(f"Feature importance: {feature_importance}")

    # Reduce to the most important dimensions
    opt_dims = np.argsort(feature_importance)[-n_opt_dims:]  # Get indices of top n_opt_dims features
    X_train = X_train[:, opt_dims]
    logger.info(f"Selected indices: {opt_dims}")
    logger.info(f"Feature importance of selected indices: {feature_importance[opt_dims]}")
    logger.info(f"Reduced X_train shape: {X_train.shape}")

    # Obtain dimensions
    input_dim = n_opt_dims
    output_dim = y_train.shape[1] if len(y_train.shape) > 1 else 1
    hidden_dims = [1024, 512, 128]

    # Set up the model
    model = GBOModel(input_dim, hidden_dims, output_dim).to(device)
    # ------------------------------------------------------------------
    # Quick‑fix: initialise final‑layer bias so that sigmoid(0) * 5 ≈ y_mean.
    # This keeps the network in the sigmoid’s high‑gradient region and
    # prevents early saturation/collapse.
    with torch.no_grad():
        y_mean = y_train.mean()
        bias_val = torch.logit(torch.tensor(y_mean / 5.0, dtype=torch.float32))
        final_linear = model.model[-2]  # last nn.Linear before Sigmoid
        final_linear.bias.data.fill_(bias_val)
        final_linear.weight.data.zero_()   # optional: avoid initial overshoot
    logger.info(f"Initialised final-layer bias to {bias_val.item():.4f}")
    # ------------------------------------------------------------------

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Train the model
    logger.info("Start model fitting")
    start_time = time.time()
    model.train()
    for epoch in range(1000):
        for zb, yb in loader:
            zb, yb = zb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(zb)

            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}, Loss: {loss.item():.6f}")
            if loss.item() < 1e-4:
                logger.info(f"Early stopping, Loss {loss.item():.6f} is below threshold")
                break
    end_time = time.time()
    logger.info(f"Model fitting took {end_time - start_time:.2f}s to finish")

    # Save GBO model
    logger.info("\nSaving GBO model")
    ckpt = {
        'model_state_dict': model.state_dict(),
        'fi_dims': opt_dims,
        'input_dim': input_dim,
        'hidden_dims': hidden_dims,
        'output_dim': output_dim,
        'normalize_input': normalize_input,
        'normalize_output': normalize_output,
        'X_mean': X_mean if normalize_input else None,
        'X_std': X_std if normalize_input else None,
        'y_mean': y_mean if normalize_output else None,
        'y_std': y_std if normalize_output else None,
    }
    torch.save(ckpt, save_file)
    logger.info(f"\nSuccessful end of script")

    # Clean up GPU memory
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    

if __name__ == "__main__":

    print("Arrived at training")
    
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    model = train_gbo(
        logfile=args.logfile,
        data_file=args.data_file,
        save_file=args.save_file,
        device=args.device,
        normalize_input=args.normalize_input,
        normalize_output=args.normalize_output,
        n_opt_dims=args.n_opt_dims,
    )