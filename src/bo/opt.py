"""
Bayesian Optimization by optimizing the Expected Improvement (EI) using DNGO or GP as surrogate model.
This script implements a multi-start optimization strategy to find the best points in the latent space.
Source: https://github.com/janschwedhelm/master-thesis/blob/main/src/gp_opt.py
"""

import argparse
import logging
import functools
import pickle
import time

import numpy as np
from scipy.optimize import minimize
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import torch
import pytorch_lightning as pl

from src.bo.gp_model import SparseGPModel
from src.utils import sparse_subset


# Arguments
parser = argparse.ArgumentParser()
opt_group = parser.add_argument_group("BO optimization")
opt_group.add_argument("--logfile", type=str, help="file to log to", default="dngo_opt.log")
opt_group.add_argument("--seed", type=int, required=True)
opt_group.add_argument("--surrogate_type", type=str, default="GP", help="Type of surrogate model: 'GP' or 'DNGO'")
opt_group.add_argument("--surrogate_file", type=str, required=True, help="path to load pretrained surrogate model from")
opt_group.add_argument("--data_file", type=str, help="file to load data from", required=True)
opt_group.add_argument("--save_file", type=str, required=True, help="file to save results to")
opt_group.add_argument("--n_out", type=int, default=5, help="Number of points to return from optimization")
opt_group.add_argument("--n_starts", type=int, default=20, help="Number of optimization runs with different initial values")
opt_group.add_argument("--n_samples", type=int, default=10000, help="Number of samples to draw from sample distribution")
opt_group.add_argument("--sample_distribution", type=str, default="normal", help="Distribution to sample from: 'normal' or 'uniform'")
opt_group.add_argument("--opt_constraint_threshold", type=float, default=None, help="Log-density threshold for optimization constraint")
opt_group.add_argument("--opt_constraint_strategy", type=str, default="gmm_fit", help="Strategy for optimization constraint: only 'gmm_fit' is implemented")
opt_group.add_argument("--n_gmm_components", type=int, default=None, help="Number of components used for GMM fitting")
opt_group.add_argument("--sparse_out", type=bool, default=True, help="Whether to filter out duplicate outputs")
opt_group.add_argument("--opt_method", type=str, default="SLSQP", help="Optimization method to use: 'SLSQP', 'COBYLA' 'L-BFGS-B'")
opt_group.add_argument("--feature_selection", type=str, default=None, choices=["PCA", "FI"], help="Feature selection method to use: 'PCA' or 'FI'. If None, no feature selection is applied.")
opt_group.add_argument("--feature_selection_dims", type=int, default=512, help="Number of (PCA or FI) dimensions to use. If feature_selection is None, this is ignored.")


# -- Functions to calculate expected improvement ------------------ #
def _ei_tensor(x):
    """
    Convert arguments to tensor for ei calcs
    Args:
        x (np.ndarray): Input data points.
    Returns:
        torch.Tensor: Converted tensor.
    """
    if len(x.shape) == 1:
        x = x.reshape(1, -1)
    return torch.tensor(x, dtype=torch.float32)


def neg_ei(x, surrogate, fmin, check_type=True, numpy=True, surrogate_type="GP"):
    """
    Calculate the negative expected improvement (EI) for a given input x.
    Args:
        x (np.ndarray): Input data points.
        surrogate (object): Surrogate model (DNGO or GP).
        fmin (float): Minimum observed value.
        check_type (bool): Whether to check the type of x.
        numpy (bool): Whether to return numpy arrays.
        surrogate_type (str): Type of surrogate model ("GP" or "DNGO").
    Returns:
        torch.Tensor or np.ndarray: Negative expected improvement.
    """
    # Convert to tensor if needed
    if check_type:
        x = _ei_tensor(x)

    # Define standard normal
    std_normal = torch.distributions.Normal(loc=0., scale=1.)
    
    if surrogate_type=="GP":
        mu, var = surrogate.predict(x)
    elif surrogate_type=="DNGO":
        batch_size = 1000
        mu = np.zeros(shape=x.shape[0], dtype=np.float32)
        var = np.zeros(shape=x.shape[0], dtype=np.float32)
        with torch.no_grad():
            # Inference variables
            batch_size = min(x.shape[0], batch_size)

            # Collect all samples
            for idx in range(x.shape[0] // batch_size):
                # Collect fake image
                mu_temp, var_temp = surrogate.predict(x[idx*batch_size : idx*batch_size + batch_size].numpy())
                mu[idx*batch_size : idx*batch_size + batch_size] = mu_temp.astype(np.float32)
                var[idx*batch_size : idx*batch_size + batch_size] = var_temp.astype(np.float32)
            
        # Convert mu and var to tensors
        mu = torch.tensor(mu, dtype=torch.float32)
        var = torch.tensor(var, dtype=torch.float32)
    else:
        raise NotImplementedError(surrogate_type)

    # Calculate EI
    sigma = torch.sqrt(var)
    z = (fmin - mu) / sigma
    ei = (fmin - mu) * std_normal.cdf(z) + sigma * torch.exp(std_normal.log_prob(z))

    if numpy: ei = ei.detach().numpy()
    
    return -ei


def neg_ei_and_grad(x, surrogate, fmin, numpy=True, surrogate_type="GP"):
    """
    Calculate the negative expected improvement (EI) and its gradient for a given input x.
    Args:
        x (np.ndarray): Input data points.
        surrogate (object): Surrogate model (DNGO or GP).
        fmin (float): Minimum observed value.
        numpy (bool): Whether to return numpy arrays.
        surrogate_type (str): Type of surrogate model ("GP" or "DNGO").
    Returns:
        torch.Tensor or np.ndarray: Negative expected improvement
        torch.Tensor or np.ndarray: Gradient of the negative expected improvement.
    """

    # Convert to tensor
    x = _ei_tensor(x)

    # Enable gradient tracking for x
    x.requires_grad_(True)

    # Compute the negative EI
    val = neg_ei(x, surrogate, fmin, check_type=False, numpy=False, surrogate_type=surrogate_type)  

    # Compute gradients
    val.backward()

    # Access the gradient of x
    grad = x.grad  
    
    if numpy:
        return val.detach().numpy(), grad.detach().numpy()
    else:
        return val, grad


# -- Functions for optimization constraints ----------------------- #
def gmm_constraint(x, fitted_gmm, threshold):
    """
    Constraint function for GMM optimization.
    Args:
        x (np.ndarray): Input data points.
        fitted_gmm (GaussianMixture): Fitted GMM model.
        threshold (float): Log-density threshold.
    Returns:
        float: Constraint value.
    """
    return -threshold + fitted_gmm.score_samples(x.reshape(1,-1))

def bound_constraint(x, component, bound):
    """
    Constraint function for bounding the optimization variables.
    Args:
        x (np.ndarray): Input data points.
        component (int): Index of the component to constrain.
        bound (float): Bound value.
    Returns:
        float: Constraint value.
    """
    return bound - np.abs(x[component])


def robust_multi_restart_optimizer(
    func_with_grad,
    X_train,
    method="SLSQP",
    num_pts_to_return=5,
    num_starts=20,
    opt_bounds=3.,
    logger=None,
    n_samples=10000,
    sample_distribution="normal",
    opt_constraint_threshold=None,
    opt_constraint_strategy="gmm_fit",
    n_gmm_components=None,
    sparse_out=True,
    feature_selection=None,
    opt_indices=None
):
    """
    Wrapper that calls scipy's optimize function at many different start points.
    Args:
        func_with_grad (callable): Function to optimize.
        X_train (np.ndarray): Training data.
        method (str): Optimization method.
        num_pts_to_return (int): Number of points to return.
        num_starts (int): Number of optimization starts.
        opt_bounds (float): Optimization bounds.
        return_res (bool): Whether to return optimization results.
        logger (logging.Logger): Logger for debugging.
        n_samples (int): Number of samples to draw from sample distribution.
        sample_distribution (str): Distribution to sample from ("normal" or "uniform").
        opt_constraint_threshold (float): Log-density threshold for optimization constraint.
        opt_constraint_strategy (str): Strategy for optimization constraint ("gmm_fit").
        n_gmm_components (int): Number of components for GMM fitting.
        sparse_out (bool): Whether to filter out duplicate outputs.
        feature_selection (str): Feature selection method ("PCA" or "FI"). If None, no feature selection is applied.
        opt_indices (np.ndarray): Indices of features to optimize if feature_selection is applied.
    Returns:
        np.ndarray: Optimized points.
        np.ndarray: Optimized values.
    """

    # -- Prepare latent grid points ------------------------------- #
    logger.debug(f"X_train shape: {X_train.shape}")

    # Sample grid points either from normal or uniform distribution
    if sample_distribution == "uniform":
        latent_grid = np.random.uniform(low=-opt_bounds, high=opt_bounds, size=(n_samples, X_train.shape[1]))
    elif sample_distribution == "normal":
        latent_grid = np.random.normal(loc=0.0, scale=0.25, size=(n_samples, X_train.shape[1]))
    elif sample_distribution == "train_data":
        latent_grid = X_train[np.random.choice(X_train.shape[0], n_samples, replace=True)]
    else:
        raise NotImplementedError(sample_distribution)

    # Store init latent grid for analysis
    latent_grid_init = latent_grid.copy()
    logger.debug(f"latent_grid_init shape: {latent_grid_init.shape}")
    logger.debug(f"opt indices: {opt_indices}")

    # Optional dim reduction through feature selection
    if feature_selection == "PCA" or feature_selection == "FI":
        assert opt_indices is not None, "opt_indices must be provided when feature_selection is 'PCA' or 'FI'"
        latent_grid = latent_grid[:, opt_indices]
        X_train = X_train[:, opt_indices]

    # Store latent grid shape
    latent_grid_dim = latent_grid.shape[1]
    logger.debug(f"Sampling points finished.")
    logger.debug(f"latent_grid shape: {latent_grid.shape}")

    # -- Optimization constraints --------------------------------- #
    # Filter out points that are below the GMM threshold if specified
    if opt_constraint_threshold is None:
        z_valid = latent_grid
    elif opt_constraint_strategy == "gmm_fit":
        assert opt_constraint_threshold, "Please specify a log-density threshold under the GMM model if 'gmm_fit' is used as optimization constraint strategy."
        assert n_gmm_components, "Please specify number of components to use for the GMM model if 'gmm_fit' is used as optimization constraint strategy."

        # Fit GMM to the latent grid
        logger.debug(f"Fitting GMM to the latent grid.")
        gmm = GaussianMixture(n_components=n_gmm_components, random_state=0, covariance_type="full", max_iter=2000, tol=1e-3).fit(X_train)
        logger.debug(f"GMM fitted with {n_gmm_components} components. Now scoring the latent grid.")
        logdens_z_grid = gmm.score_samples(latent_grid)

        # Log the shape and stats of the log-density
        logger.debug(
            f"Mean log-density: {np.mean(logdens_z_grid):.2f}, Std log-density: {np.std(logdens_z_grid):.2f}\n" +
            f"Min log-density: {np.min(logdens_z_grid):.2f}, Max log-density: {np.max(logdens_z_grid):.2f}\n" +
            f"Median log-density: {np.median(logdens_z_grid):.2f}\n" +
            f"Percentiles of log-density: {np.percentile(logdens_z_grid, [0, 5, 10, 25, 50, 75, 90, 95, 100])}"
        )

        # Filter out points that are below the threshold
        z_valid = np.array([z for i, z in enumerate(latent_grid) if logdens_z_grid[i] > opt_constraint_threshold], dtype=np.float32)
        logger.debug(f"z_valid shape: {z_valid.shape}")

        # Reduce intial grid accordingly
        latent_grid_init = latent_grid_init[np.where(logdens_z_grid > opt_constraint_threshold)[0]]
    else:
        raise NotImplementedError(opt_constraint_strategy)
    
    logger.debug(f"Finished GMM scoring. Now starting optimization.")
    
    # Sort the valid points by acquisition function
    if method == "L-BFGS-B":
        z_valid_acq, _ = func_with_grad(z_valid)
        z_valid_prop_argsort = np.argsort(z_valid_acq.reshape(1,-1))[0]  # assuming minimization of property
    elif method == "COBYLA" or method == "SLSQP":
        z_valid_acq = func_with_grad(z_valid)
        z_valid_prop_argsort = np.argsort(z_valid_acq.reshape(1,-1))[0]  # assuming minimization of property
    else:
        raise NotImplementedError(method)

    z_valid_sorted = z_valid[z_valid_prop_argsort]
    latent_grid_init = latent_grid_init[z_valid_prop_argsort]

    # -- Optimization loop ---------------------------------------- #
    # Wrapper for functions, that handles array flattening and dtype changing
    def objective1d(v):
        return tuple([arr.ravel().astype(np.float32) for arr in func_with_grad(v)])

    # Main optimization loop
    start_time = time.time()
    num_good_results = 0
    opt_results = []
    i = 0
    while (num_good_results < num_starts) and (i < z_valid_sorted.shape[0]):
        if method == "L-BFGS-B":
            if opt_constraint_threshold is None:
                res = minimize(
                    fun=objective1d, x0=z_valid_sorted[i],
                    jac=True,
                    method=method,
                    bounds=[(-opt_bounds, opt_bounds) for _ in range(latent_grid_dim)],
                    options={'gtol': 1e-08})
            else:
                raise AttributeError("A combination of 'L-BFGS-B' and GMM-optimization constraints is not possible. Please choose 'COBYLA' or 'SLSQP' as optimization method.")

            opt_results.append(res)

            if logger is not None:
                logger.info(
                    f"Iter#{i} t={time.time()-start_time:.1f}s: val={sum(res.fun):.2e}, "
                    f"success={res.success}, msg={str(res.message.decode())}, x={res.x}, jac={res.jac}, x0={z_valid_sorted[i]}")

        elif method == "COBYLA":
            if opt_constraint_threshold is None:
                res = minimize(
                    fun=objective1d, x0=z_valid_sorted[i],
                    method=method,
                    bounds=[(-opt_bounds, opt_bounds) for _ in range(latent_grid_dim)],
                    constraints=[{"type": "ineq", "fun": bound_constraint, "args": (i, opt_bounds)} for i in range(latent_grid_dim)],
                    options={'maxiter': 1000})

            else:
                res = minimize(
                    fun=objective1d, x0=z_valid_sorted[i],
                    method=method,
                    constraints=[{"type": "ineq", "fun": gmm_constraint, "args": (gmm, opt_constraint_threshold)}],
                    options={'maxiter': 1000})

            opt_results.append(res)

            if logger is not None:
                logger.info(
                    f"Iter#{i} t={time.time()-start_time:.1f}s: val={res.fun:.2e}, "
                    f"success={res.success}, msg={str(res.message)}, x={res.x}, x0={z_valid_sorted[i]}")
        
        elif method == "SLSQP":
            if opt_constraint_threshold is None:
                res = minimize(
                    fun=objective1d, x0=z_valid_sorted[i],
                    method=method,
                    bounds=[(-opt_bounds, opt_bounds) for _ in range(latent_grid_dim)],
                    options={'maxiter': 1000, 'eps': 1e-5})

            else:
                res = minimize(
                    fun=objective1d, x0=z_valid_sorted[i],
                    method=method,
                    bounds=[(-opt_bounds, opt_bounds) for _ in range(latent_grid_dim)],
                    constraints=[{"type": "ineq", "fun": gmm_constraint, "args": (gmm, opt_constraint_threshold)}],
                    options={'maxiter': 1000, 'eps': 1e-5})

            opt_results.append(res)

            if logger is not None:
                logger.info(
                    f"Iter#{i} t={time.time()-start_time:.1f}s: val={res.fun:.2e}, "
                    f"success={res.success}, msg={str(res.message)}, x={res.x}, x0={z_valid_sorted[i]}, nit={res.nit}")

        else:
            raise NotImplementedError(method)

        if res.success or (str(res.message) == "Maximum number of function evaluations has been exceeded."):
            num_good_results += 1
        i += 1

    # -- Postprocessing of optimization results ------------------- #

    # Find best successful results and their original indices
    successful = [(i, res) for i, res in enumerate(opt_results)
                  if (res.success or (str(res.message) == "Maximum number of function evaluations has been exceeded."))]
    # Sort by objective value
    sorted_pairs = sorted(successful, key=lambda pair: pair[1].fun)
    indices = [i for i, _ in sorted_pairs]
    # Build candidate arrays in sorted order
    x_candidates = np.array([res.x for _, res in sorted_pairs])
    opt_vals_candidates = np.array([res.fun for _, res in sorted_pairs])
    # Align fixed components
    latent_grid_init = latent_grid_init[indices]

    if feature_selection == "PCA" or feature_selection == "FI":
        # Merge the fixed indices back into the candidates
        latent_pred = latent_grid_init.copy()
        latent_pred[:, opt_indices] = x_candidates
    else:
        # If no feature selection, just use the candidates as is
        latent_pred = x_candidates.copy()

    # Optionally filter out duplicate optimization results
    if sparse_out:
        latent_pred, sparse_indexes = sparse_subset(latent_pred, 0.01)
        opt_vals_candidates = opt_vals_candidates[sparse_indexes]
        latent_grid_init = latent_grid_init[sparse_indexes]

    if logger is not None:
            logger.info(f"Sampled points: {latent_pred[:num_pts_to_return]}")

    return latent_pred[:num_pts_to_return], opt_vals_candidates[:num_pts_to_return], latent_grid_init[:num_pts_to_return]


def opt_main(args):
    """
    Main function to perform Bayesian optimization with DNGO or GP
    Args:
        args (argparse.Namespace): Command line arguments.
    Returns:
        np.ndarray: Optimized points.
    """
    # -- Setup & Load Data ---------------------------------------- #
    # Set up logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.FileHandler(args.logfile))

    # Load the data
    with np.load(args.data_file, allow_pickle=True) as npz:
        X_train = npz['X_train'].astype(np.float32)
        y_train = npz['y_train'].astype(np.float32)

    # Reshape the data
    x_shape = X_train.shape[1:]
    X_train = X_train.reshape(X_train.shape[0], -1)
    y_train = y_train.reshape(y_train.shape[0])
    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"y_train shape: {y_train.shape}")

    # Load training checkpoint
    ckpt = torch.load(args.surrogate_file, weights_only=False)

    # -- Optional feature selection ------------------------------- #
    if args.feature_selection == "PCA":
        pca = PCA().set_params(n_components=ckpt['pca_components'].shape[0])
        pca.components_, pca.mean_, pca.explained_variance_ = ckpt["pca_components"], ckpt["pca_mean"], ckpt["pca_explained_variance"]

        # Transform the training data using PCA
        X_train = pca.transform(X_train)
        logger.info(f"Training data min/max: {X_train.min():.2f}/{X_train.max():.2f}, "
                    f"mean/std: {X_train.mean():.2f}/{X_train.std():.2f}")
        opt_indices = np.arange(args.feature_selection_dims)
    elif args.feature_selection == "FI":
        feature_importance = ckpt["feature_importance"]

        # Sort features by importance and select top features
        sorted_indices = np.argsort(feature_importance)[::-1]
        if args.feature_selection_dims < X_train.shape[1]:
            logger.info(f"Selecting top {args.feature_selection_dims} features based on importance")
            opt_indices = sorted_indices[:args.feature_selection_dims]

    # -- Load pretrained surrogate model -------------------------- #
    if args.surrogate_type == "GP":
        surrogate = SparseGPModel(ckpt["inducing_points"], ckpt["ard_dims"])
        surrogate.load_state_dict(ckpt["state_dict"])
        surrogate.eval()
    elif args.surrogate_type == "DNGO":
        surrogate = ckpt["model"]
    else:
        raise NotImplementedError(args.surrogate_type)
    
    # -- Run Optimization ----------------------------------------- #
    # Choose a value for fmin.
    """
    In pratice, it seems that for a very small value, the EI gradients
    are very small, so the optimization doesn't converge.
    Choosing a low-ish percentile seems to be a good comprimise.
    """
    fmin = np.percentile(y_train, 10)
    logger.info(f"Using fmin={fmin:.2f}")

    # Set optimization bounds
    opt_bounds = 10 if args.feature_selection == "PCA" else 1.
    logger.info(f"Using optimization bound of {opt_bounds}")

    # Run the optimization
    logger.info("\n### Starting optimization ### \n")

    if args.opt_method == "L-BFGS-B":
        latent_pred, ei_vals, latent_grid_init = robust_multi_restart_optimizer(
            functools.partial(neg_ei_and_grad, surrogate=surrogate, fmin=fmin, surrogate_type=args.surrogate_type),
            X_train,
            args.opt_method,
            num_pts_to_return=args.n_out,
            num_starts=args.n_starts,
            opt_bounds=opt_bounds,
            n_samples=args.n_samples,
            sample_distribution=args.sample_distribution,
            logger=logger,
            opt_constraint_threshold=args.opt_constraint_threshold,
            opt_constraint_strategy=args.opt_constraint_strategy,
            n_gmm_components=args.n_gmm_components,
            sparse_out=args.sparse_out,
            feature_selection=args.feature_selection,
            opt_indices=opt_indices,
        )
    elif args.opt_method == "COBYLA" or args.opt_method=="SLSQP":
        latent_pred, ei_vals, latent_grid_init = robust_multi_restart_optimizer(
            functools.partial(neg_ei, surrogate=surrogate, fmin=fmin, surrogate_type=args.surrogate_type),
            X_train,
            args.opt_method,
            num_pts_to_return=args.n_out,
            num_starts=args.n_starts,
            opt_bounds=opt_bounds,
            n_samples=args.n_samples,
            sample_distribution=args.sample_distribution,
            logger=logger,
            opt_constraint_threshold=args.opt_constraint_threshold,
            opt_constraint_strategy=args.opt_constraint_strategy,
            n_gmm_components=args.n_gmm_components,
            sparse_out=args.sparse_out,
            feature_selection=args.feature_selection,
            opt_indices=opt_indices,
        )
    else:
        raise NotImplementedError(args.opt_method)

    logger.info(f"Done optimization! {len(latent_pred)} results found\n\n.")

    # -- Postprocessing ------------------------------------------- #

    # Make some gp predictions in the log file
    logger.info(f"EI results: {ei_vals}")
    if args.surrogate_type == "GP":
        latent_pred_tensor = torch.tensor(latent_pred)
        mu, var = surrogate.predict(latent_pred_tensor)
    elif args.surrogate_type == "DNGO":
        mu, var = surrogate.predict(latent_pred[:, opt_indices])
    logger.info(f"mu at points: {list(mu.ravel())}")
    logger.info(f"var at points: {list(var.ravel())}")

    # Optionally invert PCA
    if args.feature_selection == "PCA":
        latent_pred = pca.inverse_transform(latent_pred)
        latent_grid_init = pca.inverse_transform(latent_grid_init)
        logger.info(f"latent_pred shape after inverse PCA transform: {latent_pred.shape}")

    # Ensure datatype is float32
    latent_pred = np.array(latent_pred, dtype=np.float32)
    latent_grid_init = np.array(latent_grid_init, dtype=np.float32)

    # Reshape to original shape
    latent_pred = latent_pred.reshape(-1, *x_shape)
    latent_grid_init = latent_grid_init.reshape(-1, *x_shape)
    logger.info(f"latent_pred shape after reshape: {latent_pred.shape}")

    # Save results
    np.savez_compressed(
        args.save_file,
        z_init=latent_grid_init,
        z_opt=latent_pred
    )
    
    logger.info("\nEnd of Script.")

    return latent_pred


if __name__ == "__main__":
    args = parser.parse_args()
    pl.seed_everything(args.seed)
    opt_main(args)