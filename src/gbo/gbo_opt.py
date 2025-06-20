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
from sklearn.decomposition import PCA

from src.metrics.feature_importance import compute_feature_importance_from_data, compute_feature_importance_from_model
from src.gbo.gbo_model import GBOModel
from src.utils import zero_mean_unit_var_normalization, zero_mean_unit_var_denormalization

# Arguments
parser = argparse.ArgumentParser()
gbo_opt_group = parser.add_argument_group("GBO optimization")
gbo_opt_group.add_argument("--logfile", type=str, help="file to log to", default="gbo_opt.log")
gbo_opt_group.add_argument("--seed", type=int, required=True)
gbo_opt_group.add_argument("--model_file", type=str, help="file to load model from", required=True)
gbo_opt_group.add_argument("--save_file", type=str, required=True, help="path to save results to")
gbo_opt_group.add_argument("--data_file", type=str, help="file to load data from", required=True)
gbo_opt_group.add_argument("--n_starts", type=int, default=1, help="number of random starts")
gbo_opt_group.add_argument("--n_out", type=int, default=1, help="number of outputs")
gbo_opt_group.add_argument("--sample_distribution", type=str, default="normal", help="distribution to sample from (normal, uniform or train data)")
gbo_opt_group.add_argument("--device", type=str, default="cpu", help="device to use for training (cpu or cuda)")
gbo_opt_group.add_argument("--feature_selection", type=str, default=None, choices=["PCA", "FI"], help="Feature selection method to use: 'PCA' or 'FI'. If None, no feature selection is applied.")
gbo_opt_group.add_argument("--feature_selection_dims", type=int, default=512, help="Number of (PCA or FI) dimensions to use. If feature_selection is None, this is ignored.")


def opt_gbo(
    logfile,
    model_file,
    save_file,
    data_file,
    n_starts=1,
    n_out=1,
    sample_distribution="normal",
    device="cpu",
    feature_selection=None,
    feature_selection_dims=512,
):
    """
    Optimize the GBO model in the latent space.
    Args:
        logfile (str): Path to the log file.
        model_file (str): Path to the model file.
        save_file (str): Path to save the optimized latent variables.
        data_file (str): Path to the data file for training data distribution sampling.
        n_starts (int): Number of random starts for optimization.
        n_out (int): Number of outputs to return.
        sample_distribution (str): Distribution to sample from ("normal", "uniform", or "train_data").
        device (str): Device to use for training ('cpu' or 'cuda').
        feature_selection (str): Feature selection method to use ('PCA' or 'FI'). If None, no feature selection is applied.
        feature_selection_dims (int): Number of dimensions to use for feature selection. Ignored if
    Returns:
        np.ndarray: Optimized latent variables.
    """

    # -- Setup & Load Data ---------------------------------------- #    
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

    # Load train data
    with np.load(data_file, allow_pickle=True) as npz:
        X_train = npz['X_train'].astype(np.float32)
    x_shape = X_train.shape[1:]
    logger.info(f"X_train shape: {X_train.shape}")

    # Normalize inputs
    if normalize_input:
        logger.info("Normalizing input data")
        X_train = zero_mean_unit_var_normalization(X_train, X_mean, X_std)
        logger.debug(f"X_train stats after normalization: mean {X_train.mean()}, std {X_train.std()}, min {X_train.min()}, max {X_train.max()}")

    # -- Optional feature selection ------------------------------- #
    if feature_selection == "PCA":
        pca = PCA().set_params(n_components=ckpt['pca_components'].shape[0])
        pca.components_, pca.mean_, pca.explained_variance_ = ckpt["pca_components"], ckpt["pca_mean"], ckpt["pca_explained_variance"]

        # Transform the training data using PCA
        X_train = pca.transform(X_train)
        logger.info(f"Training data PCA min/max: {X_train.min():.2f}/{X_train.max():.2f}, "
                    f"mean/std: {X_train.mean():.2f}/{X_train.std():.2f}")
        opt_indices = np.arange(feature_selection_dims)
    elif feature_selection == "FI":
        feature_importance = ckpt["feature_importance"]

        # Sort features by importance and select top features
        sorted_indices = np.argsort(feature_importance)[::-1]
        if feature_selection_dims < X_train.shape[1]:
            logger.info(f"Selecting top {feature_selection_dims} features based on importance")
            opt_indices = sorted_indices[:feature_selection_dims]

    # -- Initialize latent grid ----------------------------------- #
    if sample_distribution == "train_data":
        logger.info("Sampling from training data distribution")
        indices = np.random.choice(X_train.shape[0], size=n_starts)
        latent_grid = torch.tensor(X_train[indices], device=device, dtype=torch.float32)

    elif sample_distribution == "uniform":
        logger.info("Sampling from uniform distribution")
        latent_grid = torch.rand(n_starts, X_train.shape[1], device=device)

    elif sample_distribution == "normal":
        logger.info("Sampling from normal distribution")
        latent_grid = torch.randn(n_starts, ckpt['input_dim'])

    else:
        raise ValueError(f"Unknown sample_distribution: {sample_distribution}")

    # Store init latent grid for analysis
    latent_grid_init = latent_grid.cpu().numpy().copy()
    logger.debug(f"latent_grid_init shape: {latent_grid_init.shape}")

    # Optional dim reduction through feature selection
    if feature_selection == "PCA" or feature_selection == "FI":
        logger.debug(f"opt indices: {opt_indices}")
        assert opt_indices is not None, "opt_indices must be provided when feature_selection is 'PCA' or 'FI'"
        latent_grid = latent_grid[:, opt_indices]
        X_train = X_train[:, opt_indices]
    logger.debug(f"Sampling points finished.")

    # -- Setup model ---------------------------------------------- #
    model = GBOModel(
        input_dim=ckpt['input_dim'],
        hidden_dims=ckpt['hidden_dims'],
        output_dim=ckpt['output_dim'],
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # -- Optimization loop ---------------------------------------- #
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

        # Logging
        logger.info(f"Iter#{i} steps={step} t={time.time() - start_time:.2f}s val={y:.2f}")
        
        # Append the results
        list_z.append(z)
        list_y.append(y)

    # -- Post-processing ------------------------------------------ #

    # Stack into arrays
    latent_arr = np.vstack(list_z)
    y_arr = np.vstack(list_y)

    # Merge fixed dimensions if feature selection was applied
    if feature_selection == "PCA" or feature_selection == "FI":
        latent_pred = latent_grid_init.copy()
        latent_pred[:, opt_indices] = latent_arr
    else:
        latent_pred = latent_arr.copy()

    # Sort y descending
    sorted_indices = np.argsort(y_arr, axis=0).flatten()[::-1]
    top_indices = sorted_indices[:n_out]

    # Filter top n_out
    latent_pred = latent_pred[top_indices]
    latent_grid_init = latent_grid_init[top_indices]

    # Ensure the output is float32
    latent_pred = latent_pred.astype(np.float32)
    latent_grid_init = latent_grid_init.astype(np.float32)

    # Optionally invert PCA transformation
    if feature_selection == "PCA":
        latent_pred = pca.inverse_transform(latent_pred)
        latent_grid_init = pca.inverse_transform(latent_grid_init)
        logger.info(f"latent_pred shape after inverse PCA transform: {latent_pred.shape}")

    # Denormalize input and output if necessary
    if normalize_input:
        latent_pred = zero_mean_unit_var_denormalization(latent_pred, X_mean, X_std)
        latent_grid_init = zero_mean_unit_var_denormalization(latent_grid_init, X_mean, X_std)
    if normalize_output:
        y_arr = zero_mean_unit_var_denormalization(y_arr, y_mean, y_std)

    # Ensure datatype is float32
    latent_pred = latent_pred.astype(np.float32)
    latent_grid_init = latent_grid_init.astype(np.float32)
    y_arr = y_arr.astype(np.float32)

    # Reshape to original shape
    latent_pred = latent_pred.reshape(-1, *x_shape)
    latent_grid_init = latent_grid_init.reshape(-1, *x_shape)

    logger.info(f"latent_pred shape: {latent_pred.shape}, latent_grid_init shape: {latent_grid_init.shape}")

    # Save the results
    logger.info("Saving results")
    np.savez_compressed(
        save_file,
        z_opt=latent_pred,
        z_init=latent_grid_init,
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
        save_file=args.save_file,
        data_file=args.data_file,
        n_starts=args.n_starts,
        n_out=args.n_out,
        sample_distribution=args.sample_distribution,
        device=args.device,
        feature_selection=args.feature_selection,
        feature_selection_dims=args.feature_selection_dims,
    )