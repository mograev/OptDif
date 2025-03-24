import functools
import argparse
import numpy as np
from scipy import stats

# Weighting functions
class DataWeighter:
    
    weight_types = ["uniform", "rank", "dbas", "fb", "rwr", "cem-pi"]

    def __init__(self, hparams):

        if hparams.weight_type in ["uniform", "fb"]:
            self.weighting_function = DataWeighter.uniform_weights
        elif hparams.weight_type == "rank":
            self.weighting_function = functools.partial(
                DataWeighter.rank_weights, k_val=hparams.rank_weight_k
            )

        # Most other implementations are from:
        # https://github.com/dhbrookes/CbAS/blob/master/src/optimization_algs.py
        elif hparams.weight_type == "dbas":
            self.weighting_function = functools.partial(
                DataWeighter.dbas_weights,
                quantile=hparams.weight_quantile,
                noise=hparams.dbas_noise,
            )
        elif hparams.weight_type == "rwr":
            self.weighting_function = functools.partial(
                DataWeighter.rwr_weights, alpha=hparams.rwr_alpha
            )
        elif hparams.weight_type == "cem-pi":
            self.weighting_function = functools.partial(
                DataWeighter.cem_pi_weights, quantile=hparams.weight_quantile
            )

        else:
            raise NotImplementedError

        self.weight_quantile = hparams.weight_quantile
        self.weight_type = hparams.weight_type

    @staticmethod
    def normalize_weights(weights: np.array):
        """ Normalizes the given weights """
        return weights / np.mean(weights)

    @staticmethod
    def reduce_weight_variance(weights: np.array, data: np.array):
        """ Reduces the variance of the given weights via data replication """

        weights_new = []
        data_new = []
        for w, d in zip(weights, data):
            if w == 0.0:
                continue
            while w > 1:
                weights_new.append(1.0)
                data_new.append(d)
                w -= 1
            weights_new.append(w)
            data_new.append(d)

        return np.array(weights_new), np.array(data_new)

    @staticmethod
    def uniform_weights(properties: np.array):
        return np.ones_like(properties)

    # computes ranks from paper
    @staticmethod
    def rank_weights(properties: np.array, k_val: float):
        """
        Calculates rank weights assuming maximization.
        Weights are not normalized.
        """
        if np.isinf(k_val):
            return np.ones_like(properties)
        ranks = np.argsort(np.argsort(-1 * properties))
        weights = 1.0 / (k_val * len(properties) + ranks)
        return weights

    @staticmethod
    def dbas_weights(properties: np.array, quantile: float, noise: float):
        y_star = np.quantile(properties, quantile)
        if np.isclose(noise, 0):
            weights = (properties >= y_star).astype(float)
        else:
            weights = stats.norm.sf(y_star, loc=properties, scale=noise)
        return weights

    @staticmethod
    def cem_pi_weights(properties: np.array, quantile: float):

        # Find quantile cutoff
        cutoff = np.quantile(properties, quantile)
        weights = (properties >= cutoff).astype(float)
        return weights

    @staticmethod
    def rwr_weights(properties: np.array, alpha: float):

        # Subtract max property value for more stable calculation
        # It doesn't change the weights since they are normalized by the sum anyways
        prop_max = np.max(properties)
        weights = np.exp(alpha * (properties - prop_max))
        weights /= np.sum(weights)
        return weights

    @staticmethod
    def add_weight_args(parser: argparse.ArgumentParser):
        weight_group = parser.add_argument_group("weighting")
        weight_group.add_argument("--weight_type", type=str, choices=DataWeighter.weight_types, default="uniform")
        weight_group.add_argument("--rank_weight_k", type=float, default=None, help="k parameter for rank weighting")
        weight_group.add_argument("--weight_quantile", type=float, default=None, help="quantile argument for dbas, cem-pi cutoffs")
        weight_group.add_argument("--dbas_noise", type=float, default=None, help="noise parameter for dbas (to induce non-binary weights)")
        weight_group.add_argument("--rwr_alpha", type=float, default=None, help="alpha value for rwr")

        return parser