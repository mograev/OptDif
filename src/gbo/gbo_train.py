"""
Train a GBO model.
"""

import logging
import argparse
import pickle
import time

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

def train_gbo(
    logfile,
    data_file,
    save_file,
    device="cpu",
    normalize_input=True,
    normalize_output=True,
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

    # Obtain dimensions
    input_dim = X_train.shape[1]
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
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)

            # logger.debug(f"Predictions: {pred}, Targets: {yb}")
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
        'input_dim': input_dim,
        'hidden_dims': hidden_dims,
        'output_dim': output_dim,
        'normalize_input': normalize_input,
        'normalize_output': normalize_output,
        'X_mean': X_mean,
        'X_std': X_std,
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
    )