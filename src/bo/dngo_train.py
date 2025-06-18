"""
Training DNGO model. Supports MCMC sampling.
Source: https://github.com/janschwedhelm/master-thesis/blob/main/src/dngo/dngo_train.py
"""

import logging
import time
import pickle
import argparse

import numpy as np
import torch
import pytorch_lightning as pl
from sklearn.decomposition import PCA

from src.metrics.feature_importance import compute_feature_importance_from_data, compute_feature_importance_from_model
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
dngo_train_group.add_argument('--feature_selection', type=str, default=None, choices=["PCA", "FI"], help="Feature selection method to use: 'PCA' or 'FI'. If None, no feature selection is applied.")
dngo_train_group.add_argument('--feature_selection_dims', type=int, default=512, help="Number of (PCA or FI) dimensions to use. If feature_selection is None, this is ignored.")
dngo_train_group.add_argument('--feature_selection_model_path', type=str, default=None, help="Path to the feature selection model file.")
dngo_train_group.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help="Device to use for training ('cpu' or 'cuda').")


def dngo_train(
    logfile,
    data_file,
    save_file,
    normalize_input=False,
    normalize_output=False,
    do_mcmc=False,
    feature_selection=None,
    feature_selection_dims=512,
    feature_selection_model_path=None,
    device='cpu',
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
        feature_selection (str): Feature selection method to use ('PCA' or 'FI'). If None, no feature selection is applied.
        feature_selection_dims (int): Number of dimensions to use for feature selection. Ignored if feature_selection is None.
        feature_selection_model_path (str): Path to the feature selection model file, if applicable.
        device (str): Device to use for training ('cpu' or 'cuda').
    """

    # -- Setup & Load Data ---------------------------------------- #
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

    # -- Optional feature selection ------------------------------- #
    if feature_selection == "PCA":
        # If model path is provided, load pre-trained PCA model, else fit PCA on the training data
        if feature_selection_model_path is not None:
            logger.info(f"Loading pre-trained PCA model from {feature_selection_model_path}")
            with open(feature_selection_model_path, 'rb') as f:
                pca = pickle.load(f)
        else:
            logger.info(f"Applying PCA with {feature_selection_dims} components")
            pca = PCA()
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

    # -- Train DNGO Model ----------------------------------------- #
    # Initialize DNGO model
    model = DNGO(normalize_input=normalize_input, normalize_output=normalize_output, do_mcmc=do_mcmc)

    logger.info("Start model fitting")
    start_time = time.time()
    model.train(X_train, y_train, do_optimize=True)
    end_time = time.time()
    logger.info(f"Model fitting took {end_time - start_time:.1f}s to finish")

    # Save DNGO model
    logger.info("\nSaving DNGO model")
    ckpt = {
        'model': model,
        'normalize_input': normalize_input,
        'normalize_output': normalize_output
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
    pl.seed_everything(args.seed)
    dngo_train(args.logfile, args.data_file, args.save_file, args.normalize_input, args.normalize_output, args.do_mcmc, args.feature_selection, args.feature_selection_dims, args.feature_selection_model_path, args.device)