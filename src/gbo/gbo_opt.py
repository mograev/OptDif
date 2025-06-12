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
from src.utils import zero_mean_unit_var_normalization, zero_mean_unit_var_denormalization

# Arguments
parser = argparse.ArgumentParser()
gbo_opt_group = parser.add_argument_group("GBO optimization")
gbo_opt_group.add_argument("--logfile", type=str, help="file to log to", default="gbo_opt.log")
gbo_opt_group.add_argument("--seed", type=int, required=True)
gbo_opt_group.add_argument("--model_file", type=str, help="file to load model from", required=True)
gbo_opt_group.add_argument("--save_file", type=str, required=True, help="path to save results to")
gbo_opt_group.add_argument("--data_file", type=str, help="file to load data from", default=None)
gbo_opt_group.add_argument("--n_starts", type=int, default=1, help="number of random starts")
gbo_opt_group.add_argument("--n_out", type=int, default=1, help="number of outputs")
gbo_opt_group.add_argument("--sample_distribution", type=str, default="normal", help="distribution to sample from (normal, uniform or train data)")
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
    data_file=None,
    n_starts=1,
    n_out=1,
    sample_distribution="normal",
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

    if sample_distribution == "train_data":
        logger.info("Sampling from training data distribution")
        if data_file is None:
            raise ValueError("data_file must be provided when sample_distribution is 'train_data'")
        
        # Load the data
        with np.load(data_file, allow_pickle=True) as npz:
            X_train = npz['X_train'].astype(np.float32)
        logger.info(f"X_train shape: {X_train.shape}")

        # Normalize inputs
        if normalize_input:
            logger.info("Normalizing input data")
            X_train = zero_mean_unit_var_normalization(X_train, X_mean, X_std)
            logger.debug(f"X_train stats after normalization: mean {X_train.mean()}, std {X_train.std()}, min {X_train.min()}, max {X_train.max()}")

        # Sample from training data
        indices = np.random.choice(X_train.shape[0], size=n_starts)
        latent_grid = torch.tensor(X_train[indices], device=device, dtype=torch.float32)

    elif sample_distribution == "uniform":
        logger.info("Sampling from uniform distribution")

        # Generate latent grid of samples
        latent_grid = torch.rand(n_starts, ckpt['input_dim'], device=device)

    elif sample_distribution == "normal":
        logger.info("Sampling from normal distribution")

        # Generate latent grid of samples
        latent_grid = torch.randn(n_starts, ckpt['input_dim']) 

    else:
        raise ValueError(f"Unknown sample_distribution: {sample_distribution}")
    
    # Store initial latent grid for visualization
    latent_grid_init = latent_grid.clone().numpy()
    if normalize_input:
        latent_grid_init = zero_mean_unit_var_denormalization(latent_grid_init, X_mean, X_std)

    # Run optimization
    logger.info("\n### STARTING OPTIMIZATION ###\n")
    start_time = time.time()
    list_z = []
    list_y = []

    for i in range(n_starts):
        # Generate random input
        z = latent_grid[i].unsqueeze(0).to(device)
        logger.info(f"z={z.cpu().numpy()}")
        z.requires_grad = True

        # Optimizer setup
        optimizer = torch.optim.Adam([z], lr=1e-3)
        n_steps = 3000

        # Adaptive ALM setup
        R = z.shape[1] ** 0.5
        lam = torch.zeros(1, device=device)
        mu = torch.tensor(10.0, device=device)
        eta1, eta2 = 0.25, 0.75
        beta_down, beta_up = 0.5, 2.0
        prev_violation = float('inf')  # keep as plain Python scalar
        mu_min, mu_max = 1e-3, 1e6
        k_update  = 10 # only touch μ every k steps

        for step in range(n_steps):
            optimizer.zero_grad()
            score = model(z)

            # Compute constraint and loss at current z
            g_norm = torch.norm(z) - R           # signed constraint value
            g_plus = torch.clamp(g_norm, min=0.) # violation (≥ 0)

            # Augmented Lagrangian
            aug_lag = -score + lam * g_plus + 0.5 * mu * g_plus ** 2
            aug_lag.backward()
            optimizer.step()

            # Logging
            if step % 100 == 0:
                logger.debug(f"Score={score.item():.4f}, g_norm={g_norm.item():.4f}, " +
                             f"aug_lag={aug_lag.item():.4f}, " +
                             f"lam={lam.item():.4f}, " +
                             f"mu={mu.item():.4f}")

            # Dual & penalty update
            with torch.no_grad():
                # Current signed constraint and its violation
                g_norm = torch.norm(z) - R
                g_plus = torch.clamp(g_norm, min=0.)
                violation = g_plus.item()  # scalar ≥ 0

                # Dual update (λ := λ + μ·g⁺, stays ≥ 0 automatically)
                lam = lam + mu * g_plus

                # Penalty update (only every k steps)
                if step % k_update == 0:
                    if violation < eta1 * prev_violation:
                        # Constraint improving fast – shrink μ but keep it ≥ mu_min
                        mu = torch.clamp(mu * beta_down, min=mu_min)
                    elif violation > eta2 * prev_violation:
                        # Constraint stagnating – grow μ but cap it at mu_max
                        mu = torch.clamp(mu * beta_up, max=mu_max)

                    prev_violation = violation

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
    sorted_indices = np.argsort(y_arr, axis=0).flatten()[::-1]
    top_indices = sorted_indices[:n_out]

    # Filter top n_out
    latent_arr = latent_arr[top_indices]
    latent_grid_init = latent_grid_init[top_indices]

    # Ensure the output is float32
    latent_arr = latent_arr.astype(np.float32)
    latent_grid_init = latent_grid_init.astype(np.float32)

    # Save the results
    logger.info("Saving results")
    np.savez_compressed(
        save_file,
        z_opt=latent_arr,
        z_init=latent_grid_init,
    )
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
        data_file=args.data_file,
        n_starts=args.n_starts,
        n_out=args.n_out,
        sample_distribution=args.sample_distribution,
        device=args.device,
    )