""" Code to train DNGO, starting from some initial data """

import logging
import time
import pickle
import numpy as np
import argparse
import pytorch_lightning as pl

from src.dngo.dngo import DNGO


# Arguments
parser = argparse.ArgumentParser()
dngo_train_group = parser.add_argument_group("DNGO training")
dngo_train_group.add_argument("--logfile", type=str, help="file to log to", default="dngo_train.log")
dngo_train_group.add_argument("--seed", type=int, required=True)
dngo_train_group.add_argument("--data_file", type=str, help="file to load data from", required=True)
dngo_train_group.add_argument("--save_file", type=str, required=True, help="path to save results to")
dngo_train_group.add_argument('--normalize_input', dest="normalize_input", action="store_true")
dngo_train_group.add_argument('--normalize_output', dest="normalize_output", action="store_true")
dngo_train_group.add_argument('--do_mcmc', dest="do_mcmc", action="store_true")


def dngo_train(args):

    # Set up logger
    LOGGER = logging.getLogger()
    LOGGER.setLevel(logging.INFO)
    LOGGER.addHandler(logging.FileHandler(args.logfile))

    # Load the data
    with np.load(args.data_file, allow_pickle=True) as npz:
        X_train = npz['X_train']
        y_train = npz['y_train']
    model = DNGO(normalize_input=args.normalize_input, normalize_output=args.normalize_output, do_mcmc=args.do_mcmc)

    # Reshape the data
    X_train = X_train.reshape(X_train.shape[0], -1)
    LOGGER.info(f"X_train shape: {X_train.shape}")
    y_train = y_train.reshape(y_train.shape[0])
    LOGGER.info(f"y_train shape: {y_train.shape}")

    LOGGER.info("Start model fitting")
    start_time = time.time()
    model.train(X_train, y_train, do_optimize=True)
    end_time = time.time()
    LOGGER.info(f"Model fitting took {end_time - start_time:.1f}s to finish")

    # Save DNGO model
    LOGGER.info("\nSave DNGO model...")
    with open(args.save_file, 'wb') as outp:
        pickle.dump(model, outp, pickle.HIGHEST_PROTOCOL)
    LOGGER.info("\nSUCCESSFUL END OF SCRIPT")


if __name__ == "__main__":

    args = parser.parse_args()
    pl.seed_everything(args.seed)
    dngo_train(args)