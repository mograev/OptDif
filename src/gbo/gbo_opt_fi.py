"""
Gradient-Based Optimization in the PCA latent space of a neural network model.
Optimization a subset of dimensions, that have a large feature importance.
This script returns optimized latent variables.
"""

import argparse
import pickle
import logging
import time

from sklearn.decomposition import PCA
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
gbo_opt_group.add_argument("--data_file", type=str, help="file to load data from", required=True)
gbo_opt_group.add_argument("--save_file", type=str, required=True, help="path to save results to")
gbo_opt_group.add_argument("--n_starts", type=int, default=1, help="number of random starts")
gbo_opt_group.add_argument("--n_out", type=int, default=1, help="number of outputs")
gbo_opt_group.add_argument("--n_opt_dims", type=int, default=512, help="number of dimensions to optimize")
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
    data_file,
    save_file,
    n_starts=1,
    n_out=1,
    n_opt_dims=512,
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

    # Load the data
    with np.load(data_file, allow_pickle=True) as npz:
        X_train = npz['X_train'].astype(np.float32)
        # Flatten, but store original shape for later
        x_shape = X_train.shape[1:]
        X_train = X_train.reshape(X_train.shape[0], -1)
    logger.info(f"X_train shape: {X_train.shape}")

    # Normalize inputs
    if normalize_input:
        logger.info("Normalizing input data")
        X_train = zero_mean_unit_var_normalization(X_train, X_mean, X_std)
        logger.debug(f"X_train stats after normalization: mean {X_train.mean()}, std {X_train.std()}, min {X_train.min()}, max {X_train.max()}")

    # Sample n_starts random points (rows) from the training data
    indices = np.random.choice(X_train.shape[0], n_starts)
    X_sampled = X_train[indices]
    
    # Get important dimensions
    opt_dims = ckpt['fi_dims']

    # Run optimization
    logger.info("\n### STARTING OPTIMIZATION ###\n")
    start_time = time.time()
    list_z = []
    list_y = []

    for i in range(n_starts):
        logger.info(f"X_start: {X_sampled[i]}")
        logger.info(f"opt dims dimensionality: {X_sampled[i, opt_dims].shape}")

        # Get important dimensions of input
        z = torch.tensor(X_sampled[i, opt_dims], dtype=torch.float32, device=device).unsqueeze(0)

        # Optimizer setup
        z.requires_grad = True
        optimizer = torch.optim.Adam([z], lr=1e-3)
        n_steps = 5000

        # Adaptive ALM setup
        R = z.shape[1] ** 0.5
        lam = 0.0
        mu = 10.0
        eta1, eta2 = 0.25, 0.75
        beta_down, beta_up = 0.5, 2.0
        prev_violation = float('inf')  # keep as plain Python scalar
        mu_min, mu_max = 1e-3, 1e6
        k_update  = 10 # only touch μ every k steps

        # Early stopping parameters
        best_score = -float('inf')
        no_improve = 0
        patience = 200  # stop if no improvement in this many steps
        improve_margin = 1e-4  # minimum improvement to count as progress


        for step in range(n_steps):
            optimizer.zero_grad()
            score = model(z)

            # Check for improvement
            current = score.item()
            if current > best_score + improve_margin:
                best_score = current
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                logger.info(f"Early stopping at step {step}, no improvement for {patience} steps.")
                break

            # Compute constraint and loss at current z
            g_norm = torch.norm(z) - R   # signed constraint value
            g_plus = torch.clamp(g_norm, min=0.) # violation (≥ 0)

            # Augmented Lagrangian
            aug_lag = -score + lam * g_plus + 0.5 * mu * g_plus ** 2
            aug_lag.backward(retain_graph=True)
            optimizer.step()

            # Logging
            if step % 100 == 0:
                logger.debug(f"Step {step}: Score={score.item():.4f}, g_norm={g_norm.item():.4f}, " +
                             f"aug_lag={aug_lag.item():.4f}, lam={lam:.4f}, mu={mu:.4f}")

            # Dual & penalty update
            with torch.no_grad():
                # Current signed constraint and its violation
                g_norm = torch.norm(z) - R
                g_plus = torch.clamp(g_norm, min=0.)
                violation = g_plus.item()  # scalar ≥ 0

                # Dual update (λ := λ + μ·g⁺, stays ≥ 0 automatically)
                lam = lam + mu * violation

                # Penalty update (only every k steps)
                if step % k_update == 0:
                    if violation < eta1 * prev_violation:
                        # Constraint improving fast – shrink μ but keep it ≥ mu_min
                        mu = max(mu * beta_down, mu_min)
                    elif violation > eta2 * prev_violation:
                        # Constraint stagnating – grow μ but cap it at mu_max
                        mu = min(mu * beta_up, mu_max)

                    prev_violation = violation

        # Get function value
        with torch.no_grad():
            f_value = model(z).item()
        y_new = np.array(f_value)

        # Combine with fixed components
        X_new = X_sampled[i].copy()
        X_new[opt_dims] = z.squeeze(0).detach().cpu().numpy()

        # Denormalize input and output if necessary
        if normalize_input:
            X_new = zero_mean_unit_var_denormalization(X_new, X_mean, X_std)
        if normalize_output:
            y_new = zero_mean_unit_var_denormalization(y_new, y_mean, y_std)

        # Logging
        logger.info(f"Iter#{i} steps={step} t={time.time() - start_time:.2f}s val={y_new:.2f}")

        # Ensure datatype is float32
        X_new = X_new.astype(np.float32)
        y_new = y_new.astype(np.float32)

        # debugging
        logger.info(f"X statistics: mean {X_new.mean()}, std {X_new.std()}" +
                f", min {X_new.min()}, max {X_new.max()}")

        # Append the results
        list_z.append(X_new)
        list_y.append(y_new)

    # Stack into arrays
    latent_arr = np.vstack(list_z)
    y_arr = np.vstack(list_y)

    # Sort y descending
    sorted_indices = np.argsort(y_arr, axis=0)[::-1]
    top_indices = sorted_indices[:n_out]

    # Denormalize initial latent variables if necessary
    if normalize_input:
        X_sampled = zero_mean_unit_var_denormalization(X_sampled, X_mean, X_std)
    
    # Filter top n_out and filter X_samples accordingly
    latent_arr = latent_arr[top_indices]
    X_sampled = X_sampled[top_indices].astype(np.float32)

    # Reshape to original input shape
    latent_arr = latent_arr.reshape(-1, *x_shape)
    X_sampled = X_sampled.reshape(-1, *x_shape)

    # Save the results
    logger.info("Saving results")
    np.savez_compressed(
        save_file,
        z_opt=latent_arr,
        z_init=X_sampled,
    )

    # Clean up GPU
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    logger.info(f"Successful end of script")

    return latent_arr

if __name__ == "__main__":

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    opt_gbo(
        logfile=args.logfile,
        model_file=args.model_file,
        data_file=args.data_file,
        save_file=args.save_file,
        n_starts=args.n_starts,
        n_out=args.n_out,
        n_opt_dims=args.n_opt_dims,
        device=args.device,
    )