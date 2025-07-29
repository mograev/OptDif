"""
Train a sparse GP (variational inducing-point regression) with GPyTorch.
This is a feature-for-feature rewrite of the GPflow version found at
https://github.com/janschwedhelm/master-thesis/blob/main/src/gp_train.py
"""

import argparse
import functools
import logging
import time
import pickle
import os
import sys

import numpy as np
import torch
import gpytorch
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

sys.path.append(os.getcwd()) # Ensure the src directory is in the Python path
from src.metrics.feature_importance import compute_feature_importance_from_data, compute_feature_importance_from_model
from src.bo.gp_model import SparseGPModel

# Add jitter to the Cholesky decomposition
gpytorch.settings.cholesky_jitter._set_value(1e-4, 1e-4, 1e-4)

# Arguments
parser = argparse.ArgumentParser()
gp_train_group = parser.add_argument_group("GP training")
gp_train_group.add_argument("--logfile", type=str, default="gp_train.log")
gp_train_group.add_argument("--seed", type=int, required=True)
gp_train_group.add_argument("--device", type=str, default="cpu")
gp_train_group.add_argument("--kmeans_init", action="store_true")
gp_train_group.add_argument("--nZ", type=int)
gp_train_group.add_argument("--data_file", type=str, required=True)
gp_train_group.add_argument("--n_opt_iter", type=int, default=100000)
gp_train_group.add_argument("--save_file", type=str, required=True)
gp_train_group.add_argument("--convergence_tol", type=float, default=5e-4)
gp_train_group.add_argument("--kernel_convergence_tol", type=float, default=2.5e-2)
gp_train_group.add_argument("--no_early_stopping", dest="early_stopping", action="store_false")
gp_train_group.add_argument("--measure_freq", type=int, default=100)
gp_train_group.add_argument("--z_noise", type=float, default=None)
gp_train_group.add_argument("--learning_rate", type=float, default=3e-2)
gp_train_group.add_argument("--kernel_learning_rate", type=float, default=1e-1)
gp_train_group.add_argument('--feature_selection', type=str, default=None, choices=["PCA", "FI"], help="Feature selection method to use: 'PCA' or 'FI'. If None, no feature selection is applied.")
gp_train_group.add_argument('--feature_selection_dims', type=int, default=512, help="Number of (PCA or FI) dimensions to use. If feature_selection is None, this is ignored.")
gp_train_group.add_argument('--feature_selection_model_path', type=str, default=None, help="Path to the feature selection model file.")


# -- Utilities ---------------------------------------------------- #
def gp_performance_metrics(model, likelihood, X_train, y_train):
    """
    Compute negative ELBO, RMSE, and predictive log-likelihood on the training set.
    Args:
        model (SparseGPModel): The sparse GP model.
        likelihood (gpytorch.likelihoods.GaussianLikelihood): The likelihood function.
        X_train (Tensor): Training input data.
        y_train (Tensor): Training target data.
    Returns:
        dict: Dictionary containing the computed metrics.
    """
    model.eval()
    likelihood.eval()
    metrics = {}

    with torch.no_grad():        
        # Compute predictive distribution
        preds = likelihood(model(X_train))
        mu = preds.mean

        # Compute Root Mean Squared Error (RMSE)
        rmse = torch.sqrt(torch.mean((mu - y_train) ** 2))
        metrics["train_rmse"] = rmse

        # Compute predictive log-likelihood
        ll = preds.log_prob(y_train).mean().item()
        metrics["train_ll"] = ll

        # Training loss = negative ELBO
        out = model(X_train)
        elbo = gpytorch.mlls.VariationalELBO(
            likelihood, model, num_data=len(X_train)
        )(out, y_train)
        metrics["loss"] = -elbo.item()

    return metrics


def _format_dict(d):
    """
    Format a dictionary for logging, by rounding values to 2 decimal places
    or using scientific notation for large values.
    Args:
        d (dict): Dictionary to format.
    Returns:
        dict: Formatted dictionary.
    """
    out = {}
    for k, v in d.items():
        out[k] = f"{v:.2f}" if abs(v) < 10 else f"{v:.2e}"
    return out


# -- Training routine --------------------------------------------- #
def gp_train(
    nZ,
    data_file,
    save_file,
    logfile,
    device="cpu",
    kmeans_init=False,
    n_opt_iter=100000,
    convergence_tol=5e-4,
    kernel_convergence_tol=2.5e-2,
    early_stopping=True,
    measure_freq=100,
    z_noise=None,
    learning_rate=3e-2,
    kernel_learning_rate=1e-1,
    feature_selection=None,
    feature_selection_dims=512,
    feature_selection_model_path=None,
):
    """
    Train a sparse Gaussian Process model using GPyTorch.
    Args:
        nZ (int): Number of inducing points.
        data_file (str): Path to the training data file.
        save_file (str): Path to save the trained model.
        logfile (str): Path to the log file.
        device (str): Device to use for training ('cpu' or 'cuda').
        kmeans_init (bool): Whether to initialize inducing points using K-means.
        n_opt_iter (int): Number of optimization iterations.
        convergence_tol (float): Convergence tolerance for early stopping.
        kernel_convergence_tol (float): Convergence tolerance for kernel parameters.
        early_stopping (bool): Whether to use early stopping.
        measure_freq (int): Frequency of performance measurement during training.
        z_noise (float or None): Optional noise added to inducing points.
        learning_rate (float): Learning rate for the optimizer.
        kernel_learning_rate (float): Learning rate for the kernel optimizer.
        feature_selection (str): Feature selection method ('PCA' or 'FI'). If None, no feature selection is applied.
        feature_selection_dims (int): Number of dimensions to use for feature selection. Ignored if feature_selection is None.
        feature_selection_model_path (str): Path to the feature selection model file, if applicable.
    """
    # -- Setup & Load Data ---------------------------------------- #
    # Logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(logfile))

    torch.set_default_dtype(torch.float32)

    # Load data in 32-bit float format
    with np.load(data_file, allow_pickle=True) as npz:
        X_train = npz["X_train"].astype(np.float32).reshape(len(npz["X_train"]), -1)
        y_train = npz["y_train"].astype(np.float32).reshape(-1)

    logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    # -- Optional feature selection ------------------------------- #
    if feature_selection == "PCA":
        # If model path is provided, load pre-trained PCA model, else fit PCA on the training data
        if feature_selection_model_path is not None:
            logger.info(f"Loading pre-trained PCA model from {feature_selection_model_path}")
            with open(feature_selection_model_path, 'rb') as f:
                pca = pickle.load(f)
        else:
            logger.info(f"Applying PCA with {feature_selection_dims} components")
            pca = PCA(n_components=feature_selection_dims)
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

    # -- Initialize hyperparameters ------------------------------- #
    logger.info("Initializing hyperparameters")

    # Number of features
    D = X_train.shape[1]

    # Initialize inducing points
    if kmeans_init:
        logger.info("Running K-means")
        kmeans = MiniBatchKMeans(
            n_clusters=nZ,
            batch_size=min(10_000, len(X_train)),
            n_init=25,
        )
        t0 = time.time()
        kmeans.fit(X_train)
        logger.info(f"K-means finished in {time.time() - t0:.1f}s")
        Z = kmeans.cluster_centers_
    else:
        Z = X_train[np.random.choice(len(X_train), nZ, replace=False)]

    # Initialize kernel parameters
    log_lengthscales = 0.1 * np.random.randn(D)
    kernel_lengthscales = np.exp(log_lengthscales)
    kernel_variance = y_train.var()
    likelihood_variance = 0.01 * y_train.var()

    # Optional Z jitter
    if z_noise is not None:
        Z = Z + z_noise * np.random.randn(*Z.shape)

    # Torch tensors
    Z = torch.as_tensor(Z)
    X_train = torch.as_tensor(X_train, device=device)
    y_train = torch.as_tensor(y_train, device=device)

    # Build model + likelihood
    model = SparseGPModel(Z, ard_dims=D)
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(1e-6)
    )
    model.mean_module.constant.requires_grad = False  # mimic GPflow’s fixed mean 0

    # Initialize params
    model.covar_module.base_kernel.lengthscale = torch.as_tensor(kernel_lengthscales)
    model.covar_module.outputscale = torch.as_tensor(kernel_variance)
    likelihood.noise = torch.as_tensor(likelihood_variance)

    # Move to device
    model = model.to(device)
    likelihood = likelihood.to(device)

    # Optimizers
    kernel_params = [
        model.covar_module.base_kernel.raw_lengthscale,
        model.covar_module.raw_outputscale,
        likelihood.raw_noise,
    ]
    fast_kernel_opt = torch.optim.Adam(kernel_params, lr=kernel_learning_rate)
    full_opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(X_train))

    # Performance bookkeeping
    perf = functools.partial(gp_performance_metrics, model, likelihood,
                             X_train, y_train)
    last_metrics = perf()
    logger.info(f"Start metrics: {_format_dict(last_metrics)}")
    start_time = time.time()

    optimize_kernel_only = True
    logger.info("Beginning optimization")
    for step in range(1, n_opt_iter + 1):
        # ------------------------------------------------------ training step --
        model.train()
        likelihood.train()

        full_opt.zero_grad()
        fast_kernel_opt.zero_grad()

        output = model(X_train)
        loss = -mll(output, y_train)
        loss.backward()

        if optimize_kernel_only:
            fast_kernel_opt.step()
        else:
            full_opt.step()

        # ------------------------------------------------ progress / stopping --
        if step % measure_freq == 0:
            model.eval()
            likelihood.eval()
            metrics = perf()
            loss_change = abs(metrics["loss"] - last_metrics["loss"])
            rel_change = loss_change / max(abs(last_metrics["loss"]), 1e-10)

            logger.info(
                f"\nStep {step}: elapsed={time.time() - start_time:.1f}s "
                f"{rel_change * 100:.2f}% Δloss"
            )
            logger.info(str(_format_dict(metrics)))
            last_metrics = metrics

            # Switch from kernel-only to full optimisation
            if optimize_kernel_only and rel_change < kernel_convergence_tol:
                optimize_kernel_only = False
                logger.info("### Switching to full-parameter optimisation ###")

            # Early stopping
            if early_stopping and rel_change < convergence_tol:
                logger.info("Early stopping (converged)")
                break

        if step == n_opt_iter:
            logger.info("Reached max iterations")

    logger.info("Training complete")

    # -- Save model ----------------------------------------------- #
    logger.info("Saving model")

    ckpt = {
        "state_dict": model.state_dict(),
        "inducing_points": model.Z.cpu(),
        "ard_dims": model.covar_module.base_kernel.ard_num_dims,
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



if __name__ == "__main__":
    
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    gp_train(args.nZ, args.data_file, args.save_file, args.logfile, args.device, args.kmeans_init, args.n_opt_iter, args.convergence_tol, args.kernel_convergence_tol, args.early_stopping, args.measure_freq, args.z_noise, args.learning_rate, args.kernel_learning_rate, args.feature_selection, args.feature_selection_dims, args.feature_selection_model_path)