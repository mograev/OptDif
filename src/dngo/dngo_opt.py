""" Code to perform Bayesian Optimization with DNGO """

import argparse
import logging
import functools
import pickle
import time
from tqdm import tqdm

import numpy as np
from scipy.optimize import minimize
from sklearn.mixture import GaussianMixture
import torch
import pytorch_lightning as pl

from src.utils import sparse_subset


# Arguments
parser = argparse.ArgumentParser()
dngo_opt_group = parser.add_argument_group("DNGO optimization")
dngo_opt_group.add_argument("--logfile", type=str, help="file to log to", default="dngo_opt.log")
dngo_opt_group.add_argument("--seed", type=int, required=True)
dngo_opt_group.add_argument("--surrogate_file", type=str, required=True, help="path to load pretrained surrogate model from")
dngo_opt_group.add_argument("--data_file", type=str, help="file to load data from", required=True)
dngo_opt_group.add_argument("--save_file", type=str, required=True, help="file to save results to")
dngo_opt_group.add_argument("--n_out", type=int, default=5, help="number of optimization points to return")
dngo_opt_group.add_argument("--n_starts", type=int, default=20, help="number of optimization runs with different initial values")
dngo_opt_group.add_argument("--n_samples", type=int, default=10000, help="Number of grid points")
dngo_opt_group.add_argument("--opt_constraint_threshold", type=float, default=None, help="Log-density threshold for optimization constraint")
dngo_opt_group.add_argument("--sample_distribution", type=str, default="normal", help="Distribution which the samples are drawn from.")
dngo_opt_group.add_argument("--opt_constraint_strategy", type=str, default="gmm_fit")
dngo_opt_group.add_argument("--n_gmm_components", type=int, default=None, help="Number of components used for GMM fitting")
dngo_opt_group.add_argument("--sparse_out", type=bool, default=True)
dngo_opt_group.add_argument("--opt_method", type=str, default="SLSQP")


# Functions to calculate expected improvement
# =============================================================================
def _ei_tensor(x):
    """ convert arguments to tensor for ei calcs """
    if len(x.shape) == 1:
        x = x.reshape(1, -1)
    return torch.tensor(x, dtype=torch.float32)


def neg_ei(x, surrogate, fmin, check_type=True):
    if check_type:
        x = _ei_tensor(x)

    # Define standard normal
    std_normal = torch.distributions.Normal(loc=0., scale=1.)
    
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
    mu = mu.astype(np.float32)
    var = var.astype(np.float32)

    # Calculate EI
    sigma = torch.sqrt(torch.tensor(var))
    z = (fmin - mu) / sigma
    ei = ((fmin - mu) * std_normal.cdf(z) +
          sigma * std_normal.prob(z))
    return -ei


def neg_ei_and_grad(x, surrogate, fmin, numpy=True):

    # Convert to tensor
    x = _ei_tensor(x)

    # Enable gradient tracking for x
    x.requires_grad_(True)

    # Compute the negative EI
    val = neg_ei(x, surrogate, fmin, check_type=False)  

    # Compute gradients
    val.backward()

    # Access the gradient of x
    grad = x.grad  
    
    if numpy:
        return val.numpy(), grad.numpy()
    else:
        return val, grad


# Functions for optimization constraints
# =============================================================================
def gmm_constraint(x, fitted_gmm, threshold):
    return -threshold + fitted_gmm.score_samples(x.reshape(1,-1))


def bound_constraint(x, component, bound):
    return bound - np.abs(x[component])


def robust_multi_restart_optimizer(
        func_with_grad,
        X_train,
        method="SLSQP",
        num_pts_to_return=5,
        num_starts=20,
        use_tqdm=False,
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
    """

    # Wrapper for tensorflow functions, that handles array flattening and dtype changing
    def objective1d(v):
        if method == "L-BFGS-B":
            return tuple([arr.ravel().astype(np.float64) for arr in func_with_grad(v)])
        elif method == "COBYLA" or method == "SLSQP":
            return tuple([arr.numpy().ravel().astype(np.float64) for arr in func_with_grad(v)])

    # Sample grid points either from normal or uniform distribution
    if sample_distribution == "uniform":
        latent_grid = np.random.uniform(low=-opt_bounds, high=opt_bounds, size=(n_samples, X_train.shape[1]))
    elif sample_distribution == "normal":
        latent_grid = np.random.normal(loc=0.0, scale=1.0, size=(n_samples, X_train.shape[1]))
    else:
        raise NotImplementedError(sample_distribution)

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
        logdens_z_grid = gmm.score_samples(latent_grid)

        # Filter out points that are below the threshold
        z_valid = np.array([z for i, z in enumerate(latent_grid) if logdens_z_grid[i] > opt_constraint_threshold],
                            dtype=np.float32)
    else:
        raise NotImplementedError(opt_constraint_strategy)
        
    # Sort the valid points by acquisition function
    if method == "L-BFGS-B":
        z_valid_acq, _ = func_with_grad(z_valid)
        z_valid_prop_argsort = np.argsort(z_valid_acq.reshape(1,-1))[0]  # assuming minimization of property
    elif method == "COBYLA" or method == "SLSQP":
        z_valid_acq = func_with_grad(z_valid)
        z_valid_prop_argsort = np.argsort(z_valid_acq.numpy().reshape(1,-1))[0]  # assuming minimization of property
    else:
        raise NotImplementedError(method)

    z_valid_sorted = z_valid[z_valid_prop_argsort]

    # Main optimization loop
    start_time = time.time()
    num_good_results = 0
    if use_tqdm:
        z_valid_sorted = tqdm(z_valid_sorted)
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


def gp_opt(args):
    """ Main function to perform Bayesian optimization with DNGO """

    # Load method
    method = args.opt_method

    # Set up logger
    LOGGER = logging.getLogger()
    LOGGER.setLevel(logging.INFO)
    LOGGER.addHandler(logging.FileHandler(args.logfile))

    # Load the data
    with np.load(args.data_path, allow_pickle=True) as npz:
        X_train = npz['X_train'].astype(np.float32)
        y_train = npz['y_train'].astype(np.float32)

    # Load pretrained DNGO
    with open(args.surrogate_file, 'rb') as inp:
        surrogate = pickle.load(inp)

    # Choose a value for fmin.
    """
    In pratice, it seems that for a very small value, the EI gradients
    are very small, so the optimization doesn't converge.
    Choosing a low-ish percentile seems to be a good comprimise.
    """
    fmin = np.percentile(y_train, 10)
    LOGGER.info(f"Using fmin={fmin:.2f}")

    # Set optimization bounds
    opt_bounds = 3
    LOGGER.info(f"Using optimization bound of {opt_bounds}")

    # Run the optimization
    LOGGER.info("\n### Starting optimization ### \n")

    if method == "L-BFGS-B":
        latent_pred, ei_vals = robust_multi_restart_optimizer(
            functools.partial(neg_ei_and_grad, surrogate=surrogate, fmin=fmin),
            X_train,
            method,
            num_pts_to_return=args.n_out,
            num_starts=args.n_starts,
            opt_bounds=opt_bounds,
            n_samples=args.n_samples,
            sample_distribution=args.sample_distribution,
            logger=LOGGER,
            opt_constraint_threshold=args.opt_constraint_threshold,
            opt_constraint_strategy=args.opt_constraint_strategy,
            n_gmm_components=args.n_gmm_components,
            sparse_out=args.sparse_out
        )
    elif method == "COBYLA" or method=="SLSQP":
        latent_pred, ei_vals = robust_multi_restart_optimizer(
            functools.partial(neg_ei, surrogate=surrogate, fmin=fmin),
            X_train,
            method,
            num_pts_to_return=args.n_out,
            num_starts=args.n_starts,
            opt_bounds=opt_bounds,
            n_samples=args.n_samples,
            sample_distribution=args.sample_distribution,
            logger=LOGGER,
            opt_constraint_threshold=args.opt_constraint_threshold,
            opt_constraint_strategy=args.opt_constraint_strategy,
            n_gmm_components=args.n_gmm_components,
            sparse_out=args.sparse_out
        )
    else:
        raise NotImplementedError(method)

    LOGGER.info(f"Done optimization! {len(latent_pred)} results found\n\n.")

    # Save results
    latent_pred = np.array(latent_pred, dtype=np.float32)
    np.save(args.save_path, latent_pred)

    # Make some gp predictions in the log file
    LOGGER.info("EI results:")
    LOGGER.info(ei_vals)

    mu, var = surrogate.predict(latent_pred)
    LOGGER.info("mu at points:")
    LOGGER.info(list(mu.ravel()))
    LOGGER.info("var at points:")
    LOGGER.info(list(var.ravel()))
    
    LOGGER.info("\n\nEND OF SCRIPT!")

    return latent_pred


if __name__ == "__main__":
    args = parser.parse_args()
    pl.seed_everything(args.seed)
    gp_opt(args)