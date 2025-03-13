""" Code to perform Bayesian Optimization with sparse GP or DNGO, using pure sampling approach """

import logging
import functools
import pickle
import time
import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize
import tensorflow as tf
import tensorflow_probability as tfp
import gpflow
from sklearn.mixture import GaussianMixture
import pytorch_lightning as pl
from pyDOE2 import *

from src.utils import sparse_subset
from src.opt_scripts.opt_celeba_sngan import generate_samples
from src.torch_mimicry.nets.sngan.sngan_64 import *

# configs
gpflow.config.set_default_float(np.float32)

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--logfile",
    type=str,
    help="file to log to",
    default="gp_opt.log"
)
parser.add_argument(
    "--seed",
    type=int,
    required=True
)
parser.add_argument(
    "--surrogate_file",
    type=str,
    required=True,
    help="file to load GP hyperparameters from",
)
parser.add_argument(
    "--data_file",
    type=str,
    help="file to load data from",
    required=True
)
parser.add_argument(
    "--save_file",
    type=str,
    required=True,
    help="File to save results to"
)
parser.add_argument(
    "--n_out",
    type=int,
    default=5,
    help="Number of optimization points to return"
)
parser.add_argument(
    "--n_starts",
    type=int,
    default=20,
    help="Number of optimization runs with different initial values (does not apply for opt_method=='sampling')"
)
parser.add_argument(
    "--n_samples",
    type=int,
    default=10000,
    help="Number of grid points to choose initial optimization value from; For opt_method=='sampling', number of samples to draw."
)
parser.add_argument(
    "--sample_distribution",
    type=str,
    default="normal",
    help="Distribution which the samples are drawn from."
)
parser.add_argument(
    "--opt_constraint_threshold",
    type=float,
    default=None,
    help="Log-density threshold for optimization constraint"
)
parser.add_argument(
    "--opt_constraint_strategy",
    type=str,
    default="gmm_fit"
)
parser.add_argument(
    "--n_gmm_components",
    type=int,
    default=None,
    help="Number of components used for GMM fitting"
)
parser.add_argument(
    "--sparse_out",
    type=bool,
    default=True,
)
parser.add_argument(
    "--pretrained_model_prior",
    type=str,
    default="normal",
    help="must be 'normal' for VAE and SN-GAN, and None for RAE"
)
parser.add_argument(
    "--opt_method",
    type=str,
    default="L-BFGS-B",
)
parser.add_argument(
    "--bo_surrogate",
    type=str,
    default="GP",
)
parser.add_argument(
    "--curr_generator_file",
    type=str,
    default=None,
)
parser.add_argument(
    "--curr_discriminator_file",
    type=str,
    default=None,
)


# Functions to calculate expected improvement
# =============================================================================
def _ei_tensor(x):
    """ convert arguments to tensor for ei calcs """
    if len(x.shape) == 1:
        x = x.reshape(1, -1)
    return tf.convert_to_tensor(x, dtype=tf.float32)


def neg_ei(x, surrogate, fmin, check_type=True, surrogate_type="GP"):
    if check_type:
        x = _ei_tensor(x)

    std_normal = tfp.distributions.Normal(loc=0., scale=1.)
    if surrogate_type=="GP":
        mu, var = surrogate.predict_f(x)
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
        mu = mu.astype(np.float32)
        var= var.astype(np.float32)
    else:
        raise NotImplementedError(surrogate_type)
    sigma = tf.sqrt(var)
    z = (fmin - mu) / sigma
    ei = ((fmin - mu) * std_normal.cdf(z) +
          sigma * std_normal.prob(z))
    return -ei


def neg_ei_and_grad(x, surrogate, fmin, numpy=True, surrogate_type="GP"):
    x = _ei_tensor(x)
    with tf.GradientTape() as tape:
        tape.watch(x)
        val = neg_ei(x, surrogate, fmin, check_type=False, surrogate_type=surrogate_type)
    grad = tape.gradient(val, x)
    if numpy:
        return val.numpy(), grad.numpy()
    else:
        return val, grad


# Functions for optimization constraints
# =============================================================================
def gmm_constraint(x, fitted_gmm, threshold):
    return -threshold + fitted_gmm.score_samples(x.reshape(1,-1))


def discriminator_constraint(x, netD, netG, probability_threshold,logger):
    netG.eval()
    netD.eval()
    x = torch.tensor(x, dtype=torch.float32).reshape(1,-1)
    image = netG.forward(x.to(netG.device))
    image = image.cpu()
    netD = netD.cpu()
    proba = torch.sigmoid(netD(image)).detach().numpy()[0]
    return proba - probability_threshold


def bound_constraint(x, component, bound):
    return bound - np.abs(x[component])


def robust_multi_restart_optimizer(
        func_with_grad,
        X_train,
        method="L-BFGS-B",
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
        sparse_out=True,
        curr_generator_file=None,
        curr_discriminator_file=None
        ):
    """
    Wrapper that calls scipy's optimize function at many different start points.
    """

    # wrapper for tensorflow functions, that handles array flattening and dtype changing
    def objective1d(v):
        if method=="L-BFGS-B":
            return tuple([arr.ravel().astype(np.float64) for arr in func_with_grad(v)])
        elif method=="COBYLA" or method=="SLSQP":
            return tuple([arr.numpy().ravel().astype(np.float64) for arr in func_with_grad(v)])
    
    if opt_constraint_strategy == "discriminator":
        # Apply discriminator constraints
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if method != "sampling":
            netG = SNGANGenerator64(nz=64).to(device)
        else:
            netG = SNGANGenerator64(nz=128).to(device)
        netD = SNGANDiscriminator64().to(device)
        netG.restore_checkpoint(curr_generator_file)
        netD.restore_checkpoint(curr_discriminator_file)
        _, latent_grid, disc_prob_threshold = generate_samples(netG, netD, n_samples, opt_constraint_threshold, method, device=netG.device)
        
        if logger is not None:
            logger.info(f"Used discriminator threshold: {disc_prob_threshold}")
        
        latent_grid = np.array(latent_grid, dtype=np.float32)
    else:
        if sample_distribution == "uniform":
            latent_grid = np.random.uniform(low=-opt_bounds, high=opt_bounds, size=(n_samples, X_train.shape[1]))
        elif sample_distribution == "normal":
            latent_grid = np.random.normal(loc=0.0, scale=1.0, size=(n_samples, X_train.shape[1]))
        else:
            raise NotImplementedError(sample_distribution)
        
    if opt_constraint_threshold is None or opt_constraint_strategy == "discriminator":
        z_valid = latent_grid
    elif opt_constraint_strategy == "gmm_fit":
        if not opt_constraint_threshold:
            raise Exception("Please specify a log-density threshold under the GMM model if "
                            "'gmm_fit' is used as optimization constraint strategy.")
        if not n_gmm_components:
            raise Exception("Please specify number of components to use for the GMM model if "
                            "'gmm_fit' is used as optimization constraint strategy.")
        gmm = GaussianMixture(n_components=n_gmm_components, random_state=0, covariance_type="full", max_iter=2000, tol=1e-3).fit(X_train)
        logdens_z_grid = gmm.score_samples(latent_grid)
        
        z_valid = np.array([z for i, z in enumerate(latent_grid) if logdens_z_grid[i] > opt_constraint_threshold],
                        dtype=np.float32)
    else:    
        raise NotImplementedError(opt_constraint_strategy)

    if method == "L-BFGS-B":
        z_valid_acq, _ = func_with_grad(z_valid)
        z_valid_prop_argsort = np.argsort(z_valid_acq.reshape(1,-1))[0]  # assuming minimization of property
    elif method == "COBYLA" or method == "SLSQP" or method == "sampling":
        z_valid_acq = func_with_grad(z_valid)
        z_valid_prop_argsort = np.argsort(z_valid_acq.numpy().reshape(1,-1))[0]  # assuming minimization of property
    else:
        raise NotImplementedError(method)

    z_valid_sorted = z_valid[z_valid_prop_argsort]
    
    if method == "sampling":
        z_valid_acq_sorted = z_valid_acq.numpy()[z_valid_prop_argsort]
        if logger is not None:
            logger.info(f"Retrieved best {num_pts_to_return} samples according to the fitted acqusition function.")
            logger.info(f"Sampled points: {z_valid_sorted[:num_pts_to_return]}")
            
        return z_valid_sorted[:num_pts_to_return], z_valid_acq_sorted[:num_pts_to_return]

    # Main optimization loop
    start_time = time.time()
    num_good_results = 0
    if use_tqdm:
        z_valid_sorted = tqdm(z_valid_sorted)
    opt_results = []
    i = 0
    while (num_good_results < num_starts) and (i < z_valid_sorted.shape[0]):
        if method=="L-BFGS-B":
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

        elif method=="COBYLA":
            if opt_constraint_threshold is None:
                res = minimize(
                    fun=objective1d, x0=z_valid_sorted[i],
                    method=method,
                    bounds=[(-opt_bounds, opt_bounds) for _ in range(X_train.shape[1])],
                    constraints=[{"type": "ineq", "fun": bound_constraint, "args": (i, opt_bounds)} for i in range(X_train.shape[1])],
                    options={'maxiter': 1000})
            elif opt_constraint_strategy == "discriminator":
                # Apply discriminator constraints
                res = minimize(
                    fun=objective1d, x0=z_valid_sorted[i],
                    method=method,
                    constraints=[{"type": "ineq", "fun": discriminator_constraint, "args": (netD, netG, disc_prob_threshold, logger)}],
                    options={'maxiter': 1000})
            else:
                # Apply GMM constraints
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
            elif opt_constraint_strategy == "discriminator":
                res = minimize(
                    fun=objective1d, x0=z_valid_sorted[i],
                    method=method,
                    constraints=[{"type": "ineq", "fun": discriminator_constraint, "args": (netD, netG, disc_prob_threshold, logger)}],
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

        if res.success or (str(res.message)=="Maximum number of function evaluations has been exceeded."):
            num_good_results += 1
        i += 1

    # Potentially directly return optimization results
    if return_res:
        return opt_results

    # Find the best successful results
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


def gp_opt(surrogate_file, data_file, save_file, logfile, n_samples, sample_distribution, num_pts_to_return, num_starts, method="L-BFGS-B",
           opt_constraint_threshold=None, opt_constraint_strategy=None, n_gmm_components=None,
           sparse_out=True, pretrained_model_prior="normal", bo_surrogate="GP", curr_generator_file=None, curr_discriminator_file=None):
    """ Do optimization via GPFlow"""
    # Start optimization script
    # Set up logger
    LOGGER = logging.getLogger()
    LOGGER.setLevel(logging.INFO)
    LOGGER.addHandler(logging.FileHandler(logfile))

    # Load the data
    with np.load(data_file, allow_pickle=True) as npz:
        X_train = npz['X_train'].astype(np.float32)
        X_test = npz['X_test']
        y_train = npz['y_train'].astype(np.float32)
        y_test = npz['y_test']

    # Initialize the GP
    if bo_surrogate == "GP":
        with np.load(surrogate_file, allow_pickle=True) as npz:
            Z = npz['Z']
            kernel_lengthscales = npz['kernel_lengthscales']
            kernel_variance = npz['kernel_variance']
            likelihood_variance = npz['likelihood_variance']

        # Make the GP
        surrogate = gpflow.models.SGPR(
            data=(X_train, y_train),
            inducing_variable=Z,
            kernel=gpflow.kernels.SquaredExponential(
                lengthscales=kernel_lengthscales,
                variance=kernel_variance
            )
        )
        surrogate.likelihood.variance.assign(likelihood_variance)
    # Load pretrained DNGO
    elif bo_surrogate == "DNGO":
        with open(surrogate_file, 'rb') as inp:
            surrogate = pickle.load(inp)
    else:
        raise NotImplementedError(bo_surrogate)

    """ 
    Choose a value for fmin.
    In pratice, it seems that for a very small value, the EI gradients
    are very small, so the optimization doesn't converge.
    Choosing a low-ish percentile seems to be a good comprimise.
    """
    fmin = np.percentile(y_train, 10)
    LOGGER.info(f"Using fmin={fmin:.2f}")

    # Choose other bounds/cutoffs
    if pretrained_model_prior == "normal":
        opt_bounds = 3
    elif pretrained_model_prior is None:
        opt_bounds = np.percentile(np.abs(X_train), 99.9)  # To account for outliers
        opt_bounds *= 1.1
    else:
        raise NotImplementedError(pretrained_model_prior)

    # Run the optimization, with a mix of random and good points
    LOGGER.info("\n### Starting optimization ### \n")

    if method == "L-BFGS-B":
        latent_pred, ei_vals = robust_multi_restart_optimizer(
            functools.partial(neg_ei_and_grad, surrogate=surrogate, fmin=fmin, surrogate_type=bo_surrogate),
            X_train,
            method,
            num_pts_to_return=num_pts_to_return,
            num_starts=num_starts,
            opt_bounds=opt_bounds,
            n_samples=n_samples,
            sample_distribution=sample_distribution,
            logger=LOGGER,
            opt_constraint_threshold=opt_constraint_threshold,
            opt_constraint_strategy=opt_constraint_strategy,
            n_gmm_components=n_gmm_components,
            sparse_out=sparse_out
        )
    elif method == "COBYLA" or method == "SLSQP" or method == "sampling":
        latent_pred, ei_vals = robust_multi_restart_optimizer(
            functools.partial(neg_ei, surrogate=surrogate, fmin=fmin, surrogate_type=bo_surrogate),
            X_train,
            method,
            num_pts_to_return=num_pts_to_return,
            num_starts=num_starts,
            opt_bounds=opt_bounds,
            n_samples=n_samples,
            sample_distribution=sample_distribution,
            logger=LOGGER,
            opt_constraint_threshold=opt_constraint_threshold,
            opt_constraint_strategy=opt_constraint_strategy,
            n_gmm_components=n_gmm_components,
            sparse_out=sparse_out,
            curr_generator_file=curr_generator_file,
            curr_discriminator_file=curr_discriminator_file
        )
    else:
        raise NotImplementedError(method)

    LOGGER.info(f"Done optimization! {len(latent_pred)} results found\n\n.")

    # Save results
    latent_pred = np.array(latent_pred, dtype=np.float32)
    np.save(save_file, latent_pred)

    # Make some gp predictions in the log file
    LOGGER.info("EI results:")
    LOGGER.info(ei_vals)

    if bo_surrogate == "GP":
        mu, var = surrogate.predict_f(latent_pred)
        LOGGER.info("mu at points:")
        LOGGER.info(list(mu.numpy().ravel()))
        LOGGER.info("var at points:")
        LOGGER.info(list(var.numpy().ravel()))
    elif bo_surrogate == "DNGO":
        mu, var = surrogate.predict(latent_pred)
        LOGGER.info("mu at points:")
        LOGGER.info(list(mu.ravel()))
        LOGGER.info("var at points:")
        LOGGER.info(list(var.ravel()))
    else:
        raise NotImplementedError(bo_surrogate)
    LOGGER.info("\n\nEND OF SCRIPT!")

    return latent_pred


if __name__ == "__main__":
    args = parser.parse_args()
    pl.seed_everything(args.seed)
    gp_opt(args.surrogate_file, args.data_file, args.save_file, args.logfile, args.n_samples, args.sample_distribution, args.n_out, args.n_starts, args.opt_method,
           args.opt_constraint_threshold, args.opt_constraint_strategy,
           args.n_gmm_components, args.sparse_out, args.pretrained_model_prior, args.bo_surrogate, args.curr_generator_file, args.curr_discriminator_file)
