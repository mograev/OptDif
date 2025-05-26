"""
Train a tree-based model as surrogate model and deterministically solve resulting MIO.
This optimization works in a discrete latent space and does not use any gradient information.
Source: https://github.com/janschwedhelm/master-thesis/blob/main/src/entmoot_bo_optimization.py
"""

import logging
import time
from collections.abc import Iterable
import argparse

import numpy as np
import torch
import pytorch_lightning as pl
from entmoot import ProblemConfig
from entmoot.models.model_params import EntingParams, UncParams, TreeTrainParams, TrainParams
from entmoot.models.enting import Enting
from entmoot.optimizers.gurobi_opt import GurobiOptimizer

from src.classification.smile_classifier import SmileClassifier

    
parser = argparse.ArgumentParser()
parser.add_argument("--logfile", type=str, help="file to log to", default="entmoot_train_opt.log")
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--data_file", type=str, help="file to load data from", required=True)
parser.add_argument("--save_file", type=str, required=True, help="file to save results to")
parser.add_argument("--predictor_path", type=str, default=None, help="path to pretrained predictor to use")
parser.add_argument("--scaled_predictor", type=bool, default=False, help="whether the predictor is scaled")
parser.add_argument("--predictor_attr_file", type=str, default=None, help="path to attribute file of the predictor")
parser.add_argument("--n_starts", type=int, default=5, help="number of initial points to sample")
parser.add_argument("--n_out", type=int, default=5, help="number of optimization points to return")
parser.add_argument("--n_dim", type=int, default=512, help="number of latent dimensions")
parser.add_argument("--n_embed", type=int, default=1024, help="number of embedding points in the discrete space")


def entmoot_train_opt(
        data_file,
        save_file,
        func,
        n_dim,
        n_embed,
        logfile="entmoot_train.log",
        n_starts=5,
        n_out=5,
        random_state=None,
        tree_kwargs={'num_boost_round': 800, 'min_data_in_leaf': 20, 'max_depth': 2},
        acq_kwargs={'dist_metric': 'l1', 'acq_sense': 'exploration'},
        optimizer_kwargs={},
):
    """
    Function to perform deterministic optimization with trained tree-based model.
    Args:
        data_file (str): Path to the data file.
        save_file (str): Path to save the results.
        func (callable): Function to optimize.
        n_dim (int): Number of dimensions.
        n_embed (int): Number of embedding points in the discrete space.
        logfile (str): Path to the log file.
        n_starts (int): Number of calls to the optimizer.
        random_state (int): Random seed for reproducibility.
        tree_kwargs (dict): Parameters for the tree-based model.
        acq_kwargs (dict): Parameters for the acquisition function.
        optimizer_kwargs (dict): Parameters for the optimizer.
    Returns:
        np.ndarray: Array of optimized points.
    """
    # -- Setup & Configuration ------------------------------------ #

    # Set up logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG) # INFO
    logger.addHandler(logging.FileHandler(logfile))

    # Load the data
    with np.load(data_file, allow_pickle=True) as npz:
        X_train = npz['X_train'].astype(np.int32).astype(str) # categorical indices
        y_train = npz['y_train'].astype(np.float32)

    # Reshape the data
    X_train = X_train.reshape(X_train.shape[0], -1)
    y_train = y_train.reshape(-1, 1)
    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"y_train shape: {y_train.shape}")
    assert len(X_train) == len(y_train), "X_train and y_train should have the same length"

    # -- Setup surrogate model and optimizer ---------------------- #

    # Describe the search space
    problem_config = ProblemConfig(rnd_seed=random_state)
    for _ in range(n_dim):
        problem_config.add_feature("categorical", [str(i) for i in range(n_embed)])
    problem_config.add_min_objective()
    logger.info(f"Problem configuration: {problem_config}")

    # Tree based model parameters
    tree_train_params = TreeTrainParams(
        TrainParams(**tree_kwargs),
    )

    # Uncertainty parameters (acquisition function)
    unc_params = UncParams(**acq_kwargs)

    # Initialize the surrogate model
    enting = Enting(
        problem_config=problem_config,
        params=EntingParams(
            tree_train_params=tree_train_params,
            unc_params=unc_params,
        ),
    )

    # Perform initial fitting
    logger.info("Start initial model fitting")
    start_time = time.time()
    enting.fit(X_train, y_train)
    end_time = time.time()
    logger.info(f"Initial model fitting took {end_time - start_time:.1f}s to finish")

    # Initialize the optimizer
    optimizer = GurobiOptimizer(
        problem_config=problem_config,
        params=optimizer_kwargs
    )

    # -- Optimization Loop ---------------------------------------- #

    # Initialize empty arrays for optimization results
    X_opt = np.zeros((n_starts, n_dim), dtype=np.float32)
    y_opt = np.zeros((n_starts, 1), dtype=np.float32)

    # Keep track of the best objective value seen so far
    best_fun = float(y_train.min())

    for i in range(n_starts):
        logger.info(f"Iteration {i + 1} of {n_starts}")

        # Solve the MIP
        logger.info("Start solving MIP")
        start_time = time.time()
        res = optimizer.solve(enting)
        end_time = time.time()
        logger.info(f"Finished solving MIP in {end_time - start_time:.1f}s")
        logger.info(f"Result: {res}")

        # Extract and evaluate the solution
        X_new = np.array(res.opt_point, dtype=np.int32)
        y_new = res.opt_val
        logger.info(f"New point: {X_new}")
        logger.info(f"Objective value: {y_new}")
        
        # # Log current results
        # logger.info(f"  new point: {X_new}")
        # new_val = float(y_new)
        # if new_val <= best_fun:
        #     logger.info(f"  new point obj.: {new_val:.5f} (*)")
        #     best_fun = new_val
        # else:
        #     logger.info(f"  new point obj.: {new_val:.5f}")
        # logger.info(f"  best obj.:       {best_fun:.5f}")

        # # Store results
        # X_opt[i] = X_new
        # y_opt[i] = y_new

        # # Append to training data
        # X_train = np.vstack((X_train, X_new))
        # y_train = np.vstack((y_train, y_new.reshape(-1, 1)))

        # # Refit the surrogate model if not the last iteration
        # if i < n_starts - 1:
        #     logger.info("Refitting surrogate model")
        #     start_time = time.time()
        #     enting.fit(X_train, y_train)
        #     end_time = time.time()
        #     logger.info(f"Refitting surrogate model took {end_time - start_time:.1f}s to finish")

    # -- Save & Log Results --------------------------------------- #

    # Sort results by objective value
    sorted_indices = np.argsort(y_opt.ravel())
    X_opt = X_opt[sorted_indices]
    y_opt = y_opt[sorted_indices]
    
    # Save n_out best points
    latent_pred = X_opt[:n_out]

    # Save latent_pred to file
    np.save(save_file, latent_pred)
    
    # Make some gp predictions in the log file
    logger.info("Acquisition function results")
    mu_at_points = [m for m, _ in enting.predict(latent_pred)]
    logger.info("mu at points:")
    logger.info(f"{mu_at_points}")

    logger.info("\n\nEND OF SCRIPT!")
    
    return latent_pred


if __name__ == "__main__":
    
    args = parser.parse_args()
    pl.seed_everything(args.seed)
    
    func = SmileClassifier(
        model_path=args.predictor_path,
        attr_file=args.predictor_attr_file,
        scaled=args.scaled_predictor
    )
    
    entmoot_train_opt(
        args.data_file,
        args.save_file,
        func,
        n_dim=args.n_dim,
        n_embed=args.n_embed,
        logfile=args.logfile,
        n_starts=args.n_starts,
        n_out=args.n_out,
        random_state=args.seed,
        tree_kwargs={'num_boost_round': 800, 'min_data_in_leaf': 20, 'max_depth': 2},
        acq_kwargs={'dist_metric': 'l1', 'acq_sense': 'exploration'},
        optimizer_kwargs={'TimeLimit': 2*10, 'MIPGap': 1e-3, 'OutputFlag': 1, 'DisplayInterval': 1},
    )