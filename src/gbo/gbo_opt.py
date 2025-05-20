"""
Gradient-Based Optimization in the latent space of a neural network model.
This script returns optimized latent variables.
"""

import argparse
import pickle
import logging
import time

import numpy as np
import torch

from src.gbo.gbo_model import GBOModel
from src.utils import zero_mean_unit_var_denormalization

# Arguments
parser = argparse.ArgumentParser()
gbo_opt_group = parser.add_argument_group("GBO optimization")
gbo_opt_group.add_argument("--logfile", type=str, help="file to log to", default="gbo_opt.log")
gbo_opt_group.add_argument("--seed", type=int, required=True)
gbo_opt_group.add_argument("--model_file", type=str, help="file to load model from", required=True)
gbo_opt_group.add_argument("--save_file", type=str, required=True, help="path to save results to")
gbo_opt_group.add_argument("--n_starts", type=int, default=1, help="number of random starts")
gbo_opt_group.add_argument("--n_out", type=int, default=1, help="number of outputs")
gbo_opt_group.add_argument("--device", type=str, default="cpu", help="device to use for training (cpu or cuda)")


def diversity_penalty(z, archive, sigma=None, lam=1.0):
    """
    Repulsion term that penalises proximity to previous solutions.
    Args:
        z (tensor): current latent vector
        archive (list): earlier solutions
        sigma (float | None): kernel width; if None use 0.25·√d
        lam (float): penalty strength
    Returns:
        torch.tensor: penalty term
    """
    if not archive:
        return torch.tensor(0.0, dtype=z.dtype, device=z.device)

    d = z.shape[1]
    if sigma is None:
        # ~25 % of typical inter‑point distance for U[0,1]^d
        sigma = 0.25 * (d ** 0.5)

    # Squared Euclidean distances WITHOUT the /d scaling
    dist2 = torch.stack([(z - p).pow(2).sum() for p in archive])

    # RBF kernel; values now drop from 1 → e⁻² ≈ 0.14 when ‖z-p‖ ≈ σ√2
    return lam * torch.exp(-dist2 / (2 * sigma ** 2)).sum()

def opt_gbo(
    logfile,
    model_file,
    save_file,
    n_starts=1,
    n_out=1,
    device="cpu",
):
    
    # Set up logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG) # INFO
    logger.addHandler(logging.FileHandler(logfile))

    # Load the checkpoint
    ckpt = torch.load(model_file, weights_only=False)
    X_mean, X_std = ckpt['X_mean'], ckpt['X_std']
    y_mean, y_std = ckpt['y_mean'], ckpt['y_std']
    normalize_input = ckpt['normalize_input']
    normalize_output = ckpt['normalize_output']

    # Setup model
    model = GBOModel(
        input_dim=ckpt['input_dim'],
        hidden_dims=ckpt['hidden_dims'],
        output_dim=ckpt['output_dim'],
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Generate latent grid of samples
    latent_grid = torch.randn(n_starts, ckpt['input_dim']) 

    # Run optimization
    logger.info("\n### STARTING OPTIMIZATION ###\n")
    start_time = time.time()
    archive = []
    list_z = []
    list_y = []

    for i in range(n_starts):
        # Generate random input
        z = latent_grid[i].unsqueeze(0).to(device)
        logger.info(f"z={z.cpu().numpy()}")
        z.requires_grad = True

        # Define the optimizer
        optimizer = torch.optim.Adam([z], lr=1e-3)

        # Early stopping
        patience = 50
        tol = 1e-4
        best_loss = float('inf')
        no_improve = 0

        n_steps = 1000
        for step in range(n_steps):
            optimizer.zero_grad()
            score = model(z)
            penalty_norm = 0.01 * torch.norm(z) # ** 2
            penalty_div = diversity_penalty(z, archive, lam=3)

            if step % 100 == 0:
                logger.debug(f"Score={score.item():.4f}, Norm={penalty_norm.item():.4f}, Div={penalty_div.item():.4f}")

            # Loss function
            loss = -score + penalty_norm + penalty_div

            # Early stopping check
            if loss < best_loss - tol:
                best_loss = loss
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                break

            loss.backward()
            optimizer.step()

        # Add to archive
        with torch.no_grad():
            archive.append(z.clone().detach())

        # Get function value
        with torch.no_grad():
            f_value = model(z).item()

        # Convert to numpy
        z = z.detach().cpu().numpy()
        y = np.array(f_value)

        # Denormalize input and output if necessary
        if normalize_input:
            z = zero_mean_unit_var_denormalization(z, X_mean, X_std)
        if normalize_output:
            y = zero_mean_unit_var_denormalization(y, y_mean, y_std)

        # Logging
        logger.info(f"Iter#{i} steps={step} t={time.time() - start_time:.2f}s val={y:.2f}")

        # Ensure datatype is float32
        z = z.astype(np.float32)
        y = y.astype(np.float32)

        # debugging
        logger.info(f"x statistics: mean {z.mean()}, std {z.std()}" +
                f", min {z.min()}, max {z.max()}")
        logger.info(f"x: {z}")
        
        # Append the results
        list_z.append(z)
        list_y.append(y)

    # Stack into arrays
    latent_arr = np.vstack(list_z)
    y_arr = np.vstack(list_y)

    # Sort y descending
    sorted_indices = np.argsort(y_arr, axis=0)[::-1]
    top_indices = sorted_indices[:n_out]
    
    # Filter top n_out
    latent_arr = latent_arr[top_indices]

    # Save the results
    logger.info("Saving results")
    np.save(save_file, latent_arr)
    logger.info(f"Successful end of script")

    # Clean up GPU
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return latent_arr

if __name__ == "__main__":

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    opt_gbo(
        logfile=args.logfile,
        model_file=args.model_file,
        save_file=args.save_file,
        n_starts=args.n_starts,
        n_out=args.n_out,
        device=args.device,
    )