"""
Code for helper functions and classes.
Reduced to functions relevant for this project.
Source: https://github.com/janschwedhelm/master-thesis/blob/main/src/utils.py
"""

import numpy as np
import torch

def zero_mean_unit_var_normalization(X, mean=None, std=None):

    compute_mean_std = mean is None and std is None

    if compute_mean_std:
        if isinstance(X, torch.Tensor):
            mean = X.mean(dim=0)
            std = X.std(dim=0)
        else:
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)

    X_normalized = (X - mean) / std

    if compute_mean_std:
        return X_normalized, mean, std
    else:
        return X_normalized


def zero_mean_unit_var_denormalization(X_normalized, mean, std):
    return X_normalized * std + mean


def sparse_subset(points, r):
    """Return a maximal list of elements of points such that no pairs of
    points in the result have distance less than r.

    """
    result = []
    index_list = []
    for i, p in enumerate(points):
        if all(np.linalg.norm(p-q) >= r for q in result):
            result.append(p)
            index_list.append(i)
    return np.array(result), index_list