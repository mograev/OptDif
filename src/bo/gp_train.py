"""
Code to train a sparse GP, starting from some initial data points.
Source: https://github.com/janschwedhelm/master-thesis/blob/main/src/gp_train.py
"""

import logging
import argparse
import functools
import time

import gpflow
import tensorflow as tf
import numpy as np
from sklearn.cluster import MiniBatchKMeans

# configs
gpflow.config.set_default_float(np.float64)


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--init", action="store_true", help="flag to initialize GP hyperparameters instead of loading them")
parser.add_argument("--logfile", type=str, help="file to log to", default="gp_train.log")
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--kmeans_init", action="store_true", help="flag to use k means init")
parser.add_argument("--nZ", type=int, help="Number of incuding points to use (if initializing)")
parser.add_argument("--data_file", type=str, help="file to load data from", required=True)
parser.add_argument("--gp_file", type=str, default=None, help="file to load GP hyperparameters from, if different than data file")
parser.add_argument("--n_perf_measure", type=int, help="Number of previous inputs to include in performance metrics specially", default=0)
parser.add_argument("--n_opt_iter", type=int, default=100000, help="Number of GP optimization iters")
parser.add_argument("--save_file", type=str, required=True, help="File to save results to")
parser.add_argument("--convergence_tol", type=float, default=5e-4, help="tolerence value to stop training GP")
parser.add_argument("--kernel_convergence_tol", type=float, default=2.5e-2, help="Tolerence value to stop training kernel only")
parser.add_argument("--no_early_stopping", dest="early_stopping", action="store_false", help="Flag to turn off early stopping")
parser.add_argument("--measure_freq", type=int, default=100, help="Frequency to measure training performance")
parser.add_argument("--z_noise", type=float, default=None, help="amplitude of normal noise to add to initialized Z points (might help GP retrain)")
parser.add_argument("--learning_rate", type=float, default=3e-2, help="Learning rate for main optimizer")
parser.add_argument("--kernel_learning_rate", type=float, default=1e-1, help="Learning rate for initial kernel optimizer")
parser.add_argument('--use_test_set', dest="use_test_set", action="store_true", help="flag to use a test set for evaluating the sparse GP")


def gp_performance_metrics(gp, datasets, dataset_names):
    """
    Return several SGPR performance metrics
    Args:
        gp (gpflow.models.SGPR): GP model
        datasets (list): list of tuples (x, y) for each dataset
        dataset_names (list): list of names for each dataset
    Returns:
        metrics (dict): dictionary of performance metrics
    """
    metrics = dict(loss=gp.training_loss().numpy())

    for name, (x, y) in zip(dataset_names, datasets):
        mu, _ = gp.predict_f(x)
        mu = mu.numpy()
        se = (mu - y)**2
        rmse = np.sqrt(np.average(se))

        log_pdf = gp.predict_log_density(data=(x, y)).numpy()
        ll = np.average(log_pdf)

        metrics[f"{name}_rmse"] = rmse
        metrics[f"{name}_ll"] = ll
    return metrics


def _format_dict(d):
    """
    Nice string formatting for losses in dictionary
    Args:
        d (dict): dictionary of values
    Returns:
        d_out (dict): dictionary of formatted values
    """
    d_out = dict()
    for k, v in list(d.items()):
        if abs(v) < 10:
            d_out[k] = f"{v:.2f}"
        else:
            d_out[k] = f"{v:.2e}"
    return d_out


def gp_train(nZ, data_file, save_file, gp_file=None, init=False, logfile="gp_train.log", kmeans_init=False, n_perf_measure=0, n_opt_iter=100000, convergence_tol=5e-4, kernel_convergence_tol=2.5e-2, early_stopping=True, measure_freq=100, z_noise=None, learning_rate=3e-2, kernel_learning_rate=1e-1, use_test_set=False):
    """
    Train a sparse GP model.
    Args:
        nZ (int): Number of inducing points to use
        data_file (str): File to load data from
        save_file (str): File to save results to
        gp_file (str, optional): File to load GP hyperparameters from, if different than data file
        init (bool, optional): Flag to initialize GP hyperparameters instead of loading them
        logfile (str, optional): File to log to
        kmeans_init (bool, optional): Flag to use k means init
        n_perf_measure (int, optional): Number of previous inputs to include in performance metrics specially
        n_opt_iter (int, optional): Number of GP optimization iters
        convergence_tol (float, optional): Tolerence value to stop training GP
        kernel_convergence_tol (float, optional): Tolerence value to stop training kernel only
        early_stopping (bool, optional): Flag to turn off early stopping
        measure_freq (int, optional): Frequency to measure training performance
        z_noise (float, optional): Amplitude of normal noise to add to initialized Z points (might help GP retrain)
        learning_rate (float, optional): Learning rate for main optimizer
        kernel_learning_rate (float, optional): Learning rate for initial kernel optimizer
        use_test_set (bool, optional): Flag to use a test set for evaluating the sparse GP
    """
    # Set up logger
    LOGGER = logging.getLogger()
    LOGGER.setLevel(logging.INFO)
    LOGGER.addHandler(logging.FileHandler(logfile))

    # Load the data
    with np.load(data_file, allow_pickle=True) as npz:
        X_train = npz['X_train']
        y_train = npz['y_train']
        if use_test_set:
            X_test = npz['X_test']
            y_test = npz['y_test']

    # Initialize points
    if init:
        LOGGER.info("Initializing GP hyperparameters")
        if kmeans_init:
            LOGGER.info("Doing Kmeans init...")
            assert nZ > 0, nZ
            # Use batched k-means, since it is significantly faster...
            kmeans = MiniBatchKMeans(
                n_clusters=nZ,
                batch_size=min(10000, X_train.shape[0]),
                n_init=25
            )
            start_time = time.time()
            kmeans.fit(X_train)
            end_time = time.time()
            LOGGER.info(f"K means took {end_time-start_time:.1f}s to finish")
            Z = kmeans.cluster_centers_.copy()
        else:
            LOGGER.info("random points init of Z")
            z_ind = np.random.choice(X_train.shape[0], nZ, replace=False)
            Z = X_train[z_ind].copy()

        # Initialize kernel lengthscales to be all near 1, with randomness to reduce symmetry
        log_lengthscales = 0.1 * \
            np.random.randn(X_train.shape[1]).astype(X_train.dtype)
        kernel_lengthscales = np.exp(log_lengthscales)

        # Initialize kernel variance to the data variance
        kernel_variance = y_train.var()

        # Likelihood variance should be very small
        likelihood_variance = 0.01 * y_train.var()

    else:
        logging.info("Loading GP parameters from a file")
        hparam_file = gp_file
        if hparam_file is None:
            hparam_file = data_file
        with np.load(hparam_file, allow_pickle=True) as npz:
            Z = npz['Z']
            kernel_lengthscales = npz['kernel_lengthscales']
            kernel_variance = npz['kernel_variance']
            likelihood_variance = npz['likelihood_variance']

    # Potentially add some noise to Z
    if z_noise is not None:
        z_noise = np.random.normal(size=Z.shape).astype(np.float64)
        Z = Z + z_noise * z_noise

    # Make the GP
    gp = gpflow.models.SGPR(
        data=(X_train, y_train),
        inducing_variable=Z,
        kernel=gpflow.kernels.SquaredExponential(
            lengthscales=kernel_lengthscales,
            variance=kernel_variance
        )
    )
    gp.likelihood.variance.assign(likelihood_variance)

    # Log performance metrics
    perf_datasets = [(X_train, y_train)]
    perf_dataset_names = ["train"]
    if use_test_set:
        perf_datasets.append((X_test, y_test))
        perf_dataset_names.append("test")
    if n_perf_measure > 0:
        perf_datasets.append(
            (X_train[-n_perf_measure:], y_train[-n_perf_measure:]))
        perf_dataset_names.append(f"train_last_{n_perf_measure}")
    perf_metrics = functools.partial(
        gp_performance_metrics,
        datasets=perf_datasets,
        dataset_names=perf_dataset_names
    )

    # Record initial metrics
    last_metrics = perf_metrics(gp)
    LOGGER.info(f"Start metrics: {_format_dict(last_metrics)}")
    LOGGER.info(gpflow.utilities.tabulate_module_summary(gp))
    start_time = time.time()
    
    # Set up optimization functions
    kernel_hparams = list(gp.kernel.trainable_variables) + \
        list(gp.likelihood.trainable_variables)
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    fast_kernel_optimizer = tf.optimizers.Adam(learning_rate=kernel_learning_rate)
    training_loss = gp.training_loss_closure(compile=True)

    @tf.function
    def optimization_step():
        optimizer.minimize(training_loss, gp.trainable_variables)
    
    @tf.function
    def kernel_optimization_step():
        fast_kernel_optimizer.minimize(training_loss, kernel_hparams)

    if init:
        LOGGER.info("Starting with kernel optimization only")
        optimize_kernel_only = True
    else:
        optimize_kernel_only = False

    # Run finer optimization with Adam
    LOGGER.info("\nStarting Adam optimization")
    for step in range(1, n_opt_iter+1):
        if optimize_kernel_only:
            kernel_optimization_step()
        else:
            optimization_step()
        if step % measure_freq == 0:

            # Find performance metrics
            metrics = perf_metrics(gp)
            loss_change = abs(metrics['loss'] - last_metrics['loss'])
            loss_rel_change = loss_change / max(abs(last_metrics['loss']), 1e-10)

            # Logging and reporting
            LOGGER.info(
                f"\nStep {step}: time elapsed={time.time() - start_time:.1f}s {loss_rel_change*100:.2f}% loss change")
            LOGGER.info(str(_format_dict(metrics)))
            LOGGER.info(gpflow.utilities.tabulate_module_summary(gp))
            last_metrics = metrics

            # Potentially stop early, or switch to full optimization
            if (loss_rel_change < kernel_convergence_tol) and optimize_kernel_only:
                optimize_kernel_only = False
                LOGGER.info("### Switching to optimization  of all params! ###")
            elif loss_rel_change < convergence_tol and early_stopping:
                LOGGER.info("\nSTOPPING TRAINING EARLY DUE TO CONVERGENCE!")
                break

        # Message for end of optimization
        if step == n_opt_iter:
            LOGGER.info(f"Optimization stopping because {n_opt_iter} iterations reached")

    # Save GP parameters
    LOGGER.info("\n\nSaving GP parameters...")
    np.savez_compressed(
        save_file,
        Z=gp.inducing_variable.Z.numpy().copy(),
        kernel_lengthscales=gp.kernel.lengthscales.numpy().copy(),
        kernel_variance=gp.kernel.variance.numpy().copy(),
        likelihood_variance=gp.likelihood.variance.numpy().copy()
    )

    LOGGER.info("\n\nSUCCESSFUL END OF SCRIPT")


if __name__ == "__main__":

    args = parser.parse_args()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    gp_train(args.nZ, args.data_file, args.save_file, args.gp_file, args.init, args.logfile, args.kmeans_init, args.n_perf_measure, args.n_opt_iter, args.convergence_tol, args.kernel_convergence_tol, args.early_stopping, args.measure_freq, args.z_noise, args.learning_rate, args.kernel_learning_rate, args.use_test_set)