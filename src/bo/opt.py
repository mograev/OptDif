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


# Functions to calculate expected improvement
# =============================================================================
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


# Functions for optimization constraints
# =============================================================================
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
        return_res=False,
        logger=None,
        n_samples=10000,
        sample_distribution="normal",
        opt_constraint_threshold=None,
        opt_constraint_strategy="gmm_fit",
        n_gmm_components=None,
        sparse_out=True
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
    Returns:
        np.ndarray: Optimized points.
        np.ndarray: Optimized values.
    """
    logger.debug(f"X_train shape: {X_train.shape}")

    # Sample grid points either from normal or uniform distribution
    if sample_distribution == "uniform":
        latent_grid = np.random.uniform(low=-opt_bounds, high=opt_bounds, size=(n_samples, X_train.shape[1]))
    elif sample_distribution == "normal":
        latent_grid = np.random.normal(loc=0.0, scale=0.25, size=(n_samples, X_train.shape[1]))
    else:
        raise NotImplementedError(sample_distribution)

    logger.debug(f"latent_grid shape: {latent_grid.shape}")
    logger.debug(f"Sampled points. Now fitting GMM to the latent grid.")

    # Filter out points that are below the GMM threshold if specified
    if opt_constraint_threshold is None:
        z_valid = latent_grid
    elif opt_constraint_strategy == "gmm_fit":
        if not opt_constraint_threshold:
            raise Exception("Please specify a log-density threshold under the GMM model if "
                            "'gmm_fit' is used as optimization constraint strategy.")
        if not n_gmm_components:
            raise Exception("Please specify number of components to use for the GMM model if "
                            "'gmm_fit' is used as optimization constraint strategy.")
        
        # Fit GMM to the latent grid
        gmm = GaussianMixture(n_components=n_gmm_components, random_state=0, covariance_type="full", max_iter=2000, tol=1e-3).fit(X_train)
        logger.debug(f"GMM fitted with {n_gmm_components} components. Now scoring the latent grid.")
        logdens_z_grid = gmm.score_samples(latent_grid)
        logger.debug(f"logdens_z_grid shape: {logdens_z_grid.shape}")
        logger.debug(f"logdens_z_grid:" f"{logdens_z_grid}")
        # print stats of the log-density, including percentiles
        logger.debug(f"Mean log-density: {np.mean(logdens_z_grid):.2f}")
        logger.debug(f"Std log-density: {np.std(logdens_z_grid):.2f}")
        logger.debug(f"Min log-density: {np.min(logdens_z_grid):.2f}")
        logger.debug(f"Max log-density: {np.max(logdens_z_grid):.2f}")
        logger.debug(f"Median log-density: {np.median(logdens_z_grid):.2f}")
        logger.debug(f"Percentiles of log-density: {np.percentile(logdens_z_grid, [0, 5, 10, 25, 50, 75, 90, 95, 100])}")

        # Filter out points that are below the threshold
        z_valid = np.array([z for i, z in enumerate(latent_grid) if logdens_z_grid[i] > opt_constraint_threshold],
                            dtype=np.float32)
        logger.debug(f"z_valid shape: {z_valid.shape}")
    else:
        raise NotImplementedError(opt_constraint_strategy)
    
    logger.debug(f"Finished GMM scoring. Now starting optimization.")
    
    # Sort the valid points by acquisition function
    if method == "L-BFGS-B":
        z_valid_acq, _ = func_with_grad(z_valid)
        z_valid_prop_argsort = np.argsort(z_valid_acq.reshape(1,-1))[0]  # assuming minimization of property
    elif method == "COBYLA" or method == "SLSQP":
        z_valid_acq = func_with_grad(z_valid)
        logger.info(f"z_valid_acq shape: {z_valid_acq.shape}")
        z_valid_prop_argsort = np.argsort(z_valid_acq.reshape(1,-1))[0]  # assuming minimization of property
    else:
        raise NotImplementedError(method)

    z_valid_sorted = z_valid[z_valid_prop_argsort]

    # Wrapper for functions, that handles array flattening and dtype changing
    def objective1d(v):
        # if method == "L-BFGS-B":
        #     return tuple([arr.ravel().astype(np.float32) for arr in func_with_grad(v)])
        # elif method == "COBYLA" or method == "SLSQP":
        #     return tuple([arr.ravel().astype(np.float32) for arr in func_with_grad(v)])
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
                    bounds=[(-opt_bounds, opt_bounds) for _ in range(X_train.shape[1])],
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
                    bounds=[(-opt_bounds, opt_bounds) for _ in range(X_train.shape[1])],
                    constraints=[{"type": "ineq", "fun": bound_constraint, "args": (i, opt_bounds)} for i in range(X_train.shape[1])],
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
                    bounds=[(-opt_bounds, opt_bounds) for _ in range(X_train.shape[1])],
                    options={'maxiter': 1000, 'eps': 1e-5})

            else:
                res = minimize(
                    fun=objective1d, x0=z_valid_sorted[i],
                    method=method,
                    bounds=[(-opt_bounds, opt_bounds) for _ in range(X_train.shape[1])],
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

    # Potentially directly return optimization results
    if return_res:
        return opt_results

    # Find best successful results
    successful_results = [res for res in opt_results if (res.success or (str(res.message) == "Maximum number of function evaluations has been exceeded."))]
    sorted_results = sorted(successful_results, key=lambda r: r.fun)
    x_candidates = np.array([res.x for res in sorted_results])
    opt_vals_candidates = np.array([res.fun for res in sorted_results])

    # Optionally filter out duplicate optimization results
    if sparse_out:
        x_candidates, sparse_indexes = sparse_subset(x_candidates, 0.01)
        opt_vals_candidates = opt_vals_candidates[sparse_indexes]

    if logger is not None:
            logger.info(f"Sampled points: {x_candidates[:num_pts_to_return]}")

    return x_candidates[:num_pts_to_return], opt_vals_candidates[:num_pts_to_return]


def opt_main(args):
    """
    Main function to perform Bayesian optimization with DNGO or GP
    Args:
        args (argparse.Namespace): Command line arguments.
    Returns:
        np.ndarray: Optimized points.
    """

    # Load method
    method = args.opt_method

    # Set up logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.FileHandler(args.logfile))

    # Load the data
    with np.load(args.data_file, allow_pickle=True) as npz:
        X_train = npz['X_train'].astype(np.float32)
        y_train = npz['y_train'].astype(np.float32)

    # Reshape the data
    X_train = X_train.reshape(X_train.shape[0], -1)
    logger.info(f"X_train shape: {X_train.shape}")
    y_train = y_train.reshape(y_train.shape[0])
    logger.info(f"y_train shape: {y_train.shape}")

    # Load pretrained surrogate model
    if args.surrogate_type == "GP":
        ckpt = torch.load(args.surrogate_file)
        surrogate = SparseGPModel(ckpt["inducing_points"], ckpt["ard_dims"])
        surrogate.load_state_dict(ckpt["state_dict"])
        surrogate.eval()
    elif args.surrogate_type == "DNGO":
        with open(args.surrogate_file, 'rb') as inp:
            surrogate = pickle.load(inp)
    else:
        raise NotImplementedError(args.surrogate_type)

    # Choose a value for fmin.
    """
    In pratice, it seems that for a very small value, the EI gradients
    are very small, so the optimization doesn't converge.
    Choosing a low-ish percentile seems to be a good comprimise.
    """
    fmin = np.percentile(y_train, 10)
    logger.info(f"Using fmin={fmin:.2f}")

    # Set optimization bounds
    opt_bounds = 1
    logger.info(f"Using optimization bound of {opt_bounds}")

    # Run the optimization
    logger.info("\n### Starting optimization ### \n")

    if method == "L-BFGS-B":
        latent_pred, ei_vals = robust_multi_restart_optimizer(
            functools.partial(neg_ei_and_grad, surrogate=surrogate, fmin=fmin, surrogate_type=args.surrogate_type),
            X_train,
            method,
            num_pts_to_return=args.n_out,
            num_starts=args.n_starts,
            opt_bounds=opt_bounds,
            n_samples=args.n_samples,
            sample_distribution=args.sample_distribution,
            logger=logger,
            opt_constraint_threshold=args.opt_constraint_threshold,
            opt_constraint_strategy=args.opt_constraint_strategy,
            n_gmm_components=args.n_gmm_components,
            sparse_out=args.sparse_out
        )
    elif method == "COBYLA" or method=="SLSQP":
        latent_pred, ei_vals = robust_multi_restart_optimizer(
            functools.partial(neg_ei, surrogate=surrogate, fmin=fmin, surrogate_type=args.surrogate_type),
            X_train,
            method,
            num_pts_to_return=args.n_out,
            num_starts=args.n_starts,
            opt_bounds=opt_bounds,
            n_samples=args.n_samples,
            sample_distribution=args.sample_distribution,
            logger=logger,
            opt_constraint_threshold=args.opt_constraint_threshold,
            opt_constraint_strategy=args.opt_constraint_strategy,
            n_gmm_components=args.n_gmm_components,
            sparse_out=args.sparse_out
        )
    else:
        raise NotImplementedError(method)

    logger.info(f"Done optimization! {len(latent_pred)} results found\n\n.")

    # Ensure datatype is float32
    latent_pred = np.array(latent_pred, dtype=np.float32)

    # Make some gp predictions in the log file
    logger.info("EI results:")
    logger.info(ei_vals)
    if args.surrogate_type == "GP":
        latent_pred_tensor = torch.tensor(latent_pred)
        mu, var = surrogate.predict(latent_pred_tensor)
    elif args.surrogate_type == "DNGO":
        mu, var = surrogate.predict(latent_pred)
    logger.info("mu at points:")
    logger.info(list(mu.ravel()))
    logger.info("var at points:")
    logger.info(list(var.ravel()))
    
    logger.info("\n\nEND OF SCRIPT!")

    # Reshape to original data shape (B, 8, 8, 8)
    latent_pred = latent_pred.reshape(latent_pred.shape[0], 8, 8, 8)
    # Save results
    np.save(args.save_file, latent_pred)

    return latent_pred


if __name__ == "__main__":
    args = parser.parse_args()
    pl.seed_everything(args.seed)
    opt_main(args)