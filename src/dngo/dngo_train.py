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
parser.add_argument(
    "--logfile",
    type=str,
    help="file to log to",
    default="dngo_train.log"
)
parser.add_argument(
    "--seed",
    type=int,
    required=True
)
parser.add_argument(
    "--data_file",
    type=str,
    help="file to load data from",
    required=True
)
parser.add_argument(
    "--dngo_file",
    type=str,
    default=None,
    help="file to load DNGO hyperparameters from, if different than data file",
)
parser.add_argument(
    "--save_file",
    type=str,
    required=True,
    help="File to save results to"
)
parser.add_argument(
    '--normalize_input',
    dest="normalize_input",
    action="store_true"
)
parser.add_argument(
    '--normalize_output',
    dest="normalize_output",
    action="store_true",
)
parser.add_argument(
    '--do_mcmc',
    dest="do_mcmc",
    action="store_true",
)


def dngo_train(data_file, save_file, logfile="dngo_train.log", normalize_input=True, normalize_output=True, do_mcmc=True):
    # Set up logger
    LOGGER = logging.getLogger()
    LOGGER.setLevel(logging.INFO)
    LOGGER.addHandler(logging.FileHandler(logfile))

    # Load the data
    with np.load(data_file, allow_pickle=True) as npz:
        X_train = npz['X_train']
        y_train = npz['y_train']
    model = DNGO(normalize_input=normalize_input, normalize_output=normalize_output, do_mcmc=do_mcmc)

    logging.info("Start model fitting")
    start_time = time.time()
    model.train(X_train, y_train.reshape(y_train.shape[0]), do_optimize=True)
    end_time = time.time()
    LOGGER.info(f"Model fitting took {end_time - start_time:.1f}s to finish")

    # Save DNGO model
    LOGGER.info("\n\nSave DNGO model...")
    with open(save_file, 'wb') as outp:
        pickle.dump(model, outp, pickle.HIGHEST_PROTOCOL)
    LOGGER.info("\n\nSUCCESSFUL END OF SCRIPT")


if __name__ == "__main__":

    args = parser.parse_args()
    pl.seed_everything(args.seed)
    dngo_train(args.data_file, args.save_file, args.logfile, args.normalize_input, args.normalize_output, args.do_mcmc)

