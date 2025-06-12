"""
Training DNGO model. Supports MCMC sampling.
Source: https://github.com/janschwedhelm/master-thesis/blob/main/src/dngo/dngo_train.py
"""

import logging
import time
import pickle
import argparse

import pytorch_lightning as pl
import torch
import numpy as np
from sklearn.decomposition import PCA

from src.bo.dngo_model import DNGO


# Arguments
parser = argparse.ArgumentParser()
dngo_train_group = parser.add_argument_group("DNGO training")
dngo_train_group.add_argument("--seed", type=int, required=True)
dngo_train_group.add_argument("--logfile", type=str, help="file to log to", default="bo_train.log")
dngo_train_group.add_argument("--data_file", type=str, help="file to load data from", required=True)
dngo_train_group.add_argument("--save_file", type=str, required=True, help="path to save results to")
dngo_train_group.add_argument('--normalize_input', dest="normalize_input", action="store_true")
dngo_train_group.add_argument('--normalize_output', dest="normalize_output", action="store_true")
dngo_train_group.add_argument('--do_mcmc', dest="do_mcmc", action="store_true")
dngo_train_group.add_argument('--n_pca_components', type=int, default=512, help="number of PCA components to use")


def dngo_train(
    logfile,
    data_file,
    save_file,
    normalize_input=False,
    normalize_output=False,
    do_mcmc=False,
    n_pca_components=512,
):
    """
    Train the DNGO model.
    Args:
        logfile (str): Path to the log file.
        data_file (str): Path to the data file.
        save_file (str): Path to save the trained model.
        normalize_input (bool): Whether to normalize input data.
        normalize_output (bool): Whether to normalize output data.
        do_mcmc (bool): Whether to perform MCMC sampling.
        n_pca_components (int): Number of PCA components to use.
    """

    # Set up logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(logfile))

    # Load the data
    with np.load(data_file, allow_pickle=True) as npz:
        X_train = npz['X_train']
        y_train = npz['y_train']

    # Reshape the data
    X_train = X_train.reshape(X_train.shape[0], -1)
    y_train = y_train.reshape(y_train.shape[0])
    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"y_train shape: {y_train.shape}")

    # Perform PCA
    pca = PCA()
    logger.info("Performing PCA on input data")
    start_time = time.time()
    Z_train = pca.fit_transform(X_train)
    logger.info(f"PCA took {time.time() - start_time:.2f}s to finish")
    logger.info(f"Z_train shape: {Z_train.shape}")

    # Reduce to n_pca_components
    Z_train = Z_train[:, :n_pca_components]
    logger.info(f"Training on {n_pca_components} PCA components")

    # Initialize DNGO model
    model = DNGO(normalize_input=normalize_input, normalize_output=normalize_output, do_mcmc=do_mcmc)

    logger.info("Start model fitting")
    start_time = time.time()
    model.train(Z_train, y_train, do_optimize=True)
    end_time = time.time()
    logger.info(f"Model fitting took {end_time - start_time:.1f}s to finish")

    # Save DNGO model
    logger.info("\nSaving DNGO model")
    ckpt = {
        'model': model,
        'pca_components': pca.components_,
        'pca_mean': pca.mean_,
        'pca_explained_variance': pca.explained_variance_,
        'normalize_input': normalize_input,
        'normalize_output': normalize_output
    }
    torch.save(ckpt, save_file)
    logger.info(f"\nSuccessful end of script")


if __name__ == "__main__":

    args = parser.parse_args()
    pl.seed_everything(args.seed)
    dngo_train(args.logfile, args.data_file, args.save_file, args.normalize_input, args.normalize_output, args.do_mcmc, args.n_pca_components)