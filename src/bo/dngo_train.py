"""
Training DNGO model. Supports MCMC sampling.
Source: https://github.com/janschwedhelm/master-thesis/blob/main/src/dngo/dngo_train.py
"""

import logging
import time
import pickle
import numpy as np
import argparse
import pytorch_lightning as pl

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


def dngo_train(
    logfile,
    data_file,
    save_file,
    normalize_input=False,
    normalize_output=False,
    do_mcmc=False,
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
    """

    # Set up logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(logfile))

    # Load the data
    with np.load(data_file, allow_pickle=True) as npz:
        X_train = npz['X_train']
        y_train = npz['y_train']
    model = DNGO(normalize_input=normalize_input, normalize_output=normalize_output, do_mcmc=do_mcmc)

    # Reshape the data
    X_train = X_train.reshape(X_train.shape[0], -1)
    y_train = y_train.reshape(y_train.shape[0])
    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"y_train shape: {y_train.shape}")

    logger.info("Start model fitting")
    start_time = time.time()
    model.train(X_train, y_train, do_optimize=True)
    end_time = time.time()
    logger.info(f"Model fitting took {end_time - start_time:.1f}s to finish")

    # Save DNGO model
    logger.info("\nSave DNGO model...")
    with open(save_file, 'wb') as outp:
        pickle.dump(model, outp, pickle.HIGHEST_PROTOCOL)
    logger.info("\nSUCCESSFUL END OF SCRIPT")


if __name__ == "__main__":

    args = parser.parse_args()
    pl.seed_everything(args.seed)
    dngo_train(args.logfile, args.data_file, args.save_file, args.normalize_input, args.normalize_output, args.do_mcmc)