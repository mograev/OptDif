"""
Train a GBO model.
"""

import logging
import argparse
import pickle
import time
from itertools import cycle, islice
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.decomposition import PCA

sys.path.append(os.getcwd()) # Ensure the src directory is in the Python path
from src.metrics.feature_importance import compute_feature_importance_from_data, compute_feature_importance_from_model
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
gbo_train_group.add_argument('--feature_selection', type=str, default=None, choices=["PCA", "FI"], help="Feature selection method to use: 'PCA' or 'FI'. If None, no feature selection is applied.")
gbo_train_group.add_argument('--feature_selection_dims', type=int, default=512, help="Number of (PCA or FI) dimensions to use. If feature_selection is None, this is ignored.")
gbo_train_group.add_argument('--feature_selection_model_path', type=str, default=None, help="Path to the feature selection model file.")


def train_gbo(
    logfile,
    data_file,
    save_file,
    device="cpu",
    normalize_input=True,
    normalize_output=True,
    feature_selection=None,
    feature_selection_dims=512,
    feature_selection_model_path=None,
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
        feature_selection (str): Feature selection method to use ('PCA' or 'FI'). If None, no feature selection is applied.
        feature_selection_dims (int): Number of dimensions to use for feature selection. Ignored if feature_selection is None.
        feature_selection_model_path (str): Path to the feature selection model file
    """

    # -- Setup & Load Data ---------------------------------------- #
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
    logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

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

    # -- Optional feature selection ------------------------------- #
    if feature_selection == "PCA":
        # If model path is provided, load pre-trained PCA model, else fit PCA on the training data
        if feature_selection_model_path is not None:
            logger.info(f"Loading pre-trained PCA model from {feature_selection_model_path}")
            with open(feature_selection_model_path, 'rb') as f:
                pca = pickle.load(f)
        else:
            logger.info(f"Applying PCA with {feature_selection_dims} components")
            pca = PCA()
            pca.fit(X_train)
            logger.info(f"Transformed X_train shape after PCA: {X_train.shape}")

        # Transform the training data using PCA
        X_train = pca.transform(X_train)

        # Reduce to feature_selection_dims components
        if feature_selection_dims < X_train.shape[1]:
            logger.info(f"Reducing to {feature_selection_dims} PCA components")
            X_train = X_train[:, :feature_selection_dims]

    elif feature_selection == "FI":
        # If model path is provided, load pre-trained feature importance model, else compute feature importance
        if feature_selection_model_path is not None:
            # load state dict from file
            logger.info(f"Loading pre-trained feature importance model from {feature_selection_model_path}")
            with open(feature_selection_model_path, 'rb') as f:
                feature_importance_model = pickle.load(f)
            feature_importance = compute_feature_importance_from_model(
                feature_importance_model, X_train, device=device
            )
        else:
            logger.info(f"Computing feature importance using training data")
            # Compute feature importance from training data
            feature_importance = compute_feature_importance_from_data(
                X_train, y_train, hidden_dims=[128, 64], lr=1e-3, batch_size=64, epochs=100, device=device
            )

        # Select top feature_selection_dims features based on importance
        if feature_selection_dims < feature_importance.shape[0]:
            logger.info(f"Selecting top {feature_selection_dims} features based on importance")
            top_indices = np.argsort(feature_importance)[-feature_selection_dims:]
            X_train = X_train[:, top_indices]
        else:
            logger.info("Using all features as feature_selection_dims is greater than or equal to the number of features")

    # -- Setup GBO Model ------------------------------------------ #
    # Obtain dimensions
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1] if len(y_train.shape) > 1 else 1
    hidden_dims = [1024, 512, 128]

    # Set up the model
    model = GBOModel(input_dim, hidden_dims, output_dim).to(device)
    """Quick-fix: initialise final-layer bias so that sigmoid(0) * 5 ≈ y_mean. This keeps the network
    in the sigmoid's high-gradient region and prevents early saturation/collapse."""
    with torch.no_grad():
        y_mean = y_train.mean()
        bias_val = torch.logit(torch.tensor(y_mean / 5.0, dtype=torch.float32))
        final_linear = model.model[-2]  # last nn.Linear before Sigmoid
        final_linear.bias.data.fill_(bias_val)
        final_linear.weight.data.zero_()   # optional: avoid initial overshoot
    logger.info(f"Initialised final-layer bias to {bias_val.item():.4f}")

    # -- Train GBO Model ------------------------------------------ #
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    n_steps = 100_000

    # Early stopping parameters
    best_loss = float('inf')
    no_improve = 0
    epoch_len = len(loader)
    patience = 5 * epoch_len  # stop if no improvement for 5 epochs

    # Store best model state
    best_model_state = model.state_dict()

    # Train the model with a fixed number of steps
    logger.info("Start model fitting")
    start_time = time.time()
    model.train()
    step = 0
    for xb, yb in islice(cycle(loader), n_steps):
        step += 1
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()

        current = loss.item()
        # Early stopping & model checkpointing
        if current < best_loss:
            logger.info(f"Step {step}, Loss improved: {current:.6f} (previous best: {best_loss:.6f})")
            best_model_state = model.state_dict()  # save the best model state
            best_loss = current
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            logger.info(f"Early stopping at step {step}, no improvement for {patience} steps.")
            break

        # Periodic logging
        if step % 100 == 0:
            logger.info(f"Step {step}, Loss: {current:.6f}")

    end_time = time.time()
    logger.info(f"Model fitting took {end_time - start_time:.2f}s to finish")

    # Load the best model state
    model.load_state_dict(best_model_state)

    # -- Save GBO model ------------------------------------------- #
    logger.info("\nSaving GBO model")
    ckpt = {
        'model_state_dict': model.state_dict(),
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
    if feature_selection == "PCA":
        ckpt.update({
            'pca_components': pca.components_,
            'pca_mean': pca.mean_,
            'pca_explained_variance': pca.explained_variance_,
        })
    elif feature_selection == "FI":
        ckpt.update({
            'feature_importance': feature_importance
        })

    torch.save(ckpt, save_file)
    logger.info(f"\nSuccessful end of script")

    # Clean up GPU memory
    del model
    if device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    model = train_gbo(
        logfile=args.logfile,
        data_file=args.data_file,
        save_file=args.save_file,
        device=args.device,
        normalize_input=args.normalize_input,
        normalize_output=args.normalize_output,
        feature_selection=args.feature_selection,
        feature_selection_dims=args.feature_selection_dims,
        feature_selection_model_path=args.feature_selection_model_path,
    )