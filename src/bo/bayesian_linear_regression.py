"""
Bayesian Linear Regression model to predict the mean and variance of a target variable given input data points.
This implementation is a torch-based rewrite of the linked code.
Sources:
- https://github.com/janschwedhelm/master-thesis/blob/main/src/dngo/bayesian_linear_regression.py
- https://github.com/janschwedhelm/master-thesis/blob/main/src/dngo/base_model.py
"""

import logging
import math

import torch
from torch.distributions import Normal


def linear_basis_func(x):
    """
    Linear basis function that appends a column of ones to the input data.
    Args:
        x (torch.Tensor): Input data tensor of shape (N, D)
    Returns:
        torch.Tensor: Tensor of shape (N, D+1) containing x and the bias term
    """
    ones = torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)
    return torch.cat((x, ones), dim=1)


def quadratic_basis_func(x):
    """
    Quadratic basis function that appends a column of squared inputs and a bias term to the input data.
    Args:
        x (torch.Tensor): Input data tensor of shape (N, D)
    Returns:
        torch.Tensor: Tensor of shape (N, 2D+1) containing x^2, x, and the bias term
    """
    quad = x ** 2
    ones = torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)
    return torch.cat((quad, x, ones), dim=1)


logger = logging.getLogger(__name__)


class Prior:
    """
    Prior class for Bayesian linear regression hyperparameters.
    This class defines a prior distribution for the hyperparameters alpha and beta.
    """

    def __init__(self, device=None):
        """
        Initializes the prior for the hyperparameters alpha and beta.
        Args:
            device (torch.device): Device to use for tensors (default: None, uses CPU)
        """
        self.device = device if device is not None else torch.device('cpu')

        # Standard normal distribution for log alpha
        self._normal_alpha = Normal(
            loc=torch.tensor(0.0, device=self.device),
            scale=torch.tensor(1.0, device=self.device)
        )

        # Normal distribution for log beta (matches original implementation)
        self._normal_beta = Normal(
            loc=torch.tensor(-3.0, device=self.device),
            scale=torch.tensor(1.0, device=self.device)
        )

    def lnprob(self, theta):
        """
        Compute the log probability for theta = [log alpha, log beta].
        Args:
            theta (torch.Tensor): Tensor of shape (2,) containing log alpha and log beta
        Returns:
            torch.Tensor: Log probability of the hyperparameters
        """
        lp_alpha = self._normal_alpha.log_prob(theta[..., 0])
        lp_beta = self._normal_beta.log_prob(theta[..., 1])
        return (lp_alpha + lp_beta).sum()

    def sample_from_prior(self, n_samples):
        """
        Sample from the prior distribution for hyperparameters.
        Args:
            n_samples (int): Number of samples to draw from the prior
        Returns:
            torch.Tensor: Samples of shape (n_samples, 2) for log alpha and log beta
        """
        alpha_samples = self._normal_alpha.sample((n_samples,))
        beta_samples = self._normal_beta.sample((n_samples,))
        return torch.stack((alpha_samples, beta_samples), dim=-1)


class BayesianLinearRegression:
    """
    Bayesian Linear Regression model that predicts the mean and variance of a target variable given input data points.
    This implementation optionally uses MCMC sampling to estimate hyperparameters (alpha and beta).
    """
    def __init__(self, alpha=1., beta=1000., basis_func=linear_basis_func, prior=None, device=None):
        """
        Initializes the Bayesian Linear Regression model.
        Args:
            alpha (float): Variance of the prior for the weights w
            beta (float): Inverse of the noise, i.e., beta = 1 / sigma^2
            basis_func (function): Function to transform the input data via basis functions
            prior (Prior): Prior for alpha and beta. If None, the default prior is used
            device (torch.device): Device to use for computations (default: None, uses CPU)
        """
        self.device = device if device is not None else torch.device('cpu')

        self.alpha0 = float(alpha)
        self.beta0 = float(beta)

        self.basis_func = basis_func
        self.prior = prior if prior is not None else Prior(device=self.device)

        # Will be filled by the train method
        self.X = None       # raw input data (N, D)
        self.y = None       # target values (N)
        self.Phi = None     # basis function features (N, M)

    def marginal_log_likelihood(self, theta):
        """
        Computes the log likelihood of the data marginalised over the weights w.
        Args:
            theta (torch.Tensor): Hyperparameters alpha and beta on a log scale in a tensor of shape (..., 2)
        Returns:
            torch.Tensor: The marginal log likelihood of the data
        """
        # Unpack and convert to linear scale
        log_alpha, log_beta = theta[..., 0], theta[..., 1]
        alpha = torch.exp(log_alpha)        # weight prior precision
        beta = torch.exp(log_beta)     # noise precision

        Phi = self.Phi  # [N, M]
        t = self.y      # [N]
        N, M = Phi.shape
        I_M = torch.eye(M, device=self.device, dtype=Phi.dtype)

        # Compute posterior covariance of weights
        A = alpha * I_M + beta * (Phi.T @ Phi)  # [M, M]

        # Use Cholesky for stability when possible
        try:
            L = torch.linalg.cholesky(A)
            log_det_A = 2 * torch.sum(torch.log(torch.diag(L)))
            rhs = (Phi.T @ t).unsqueeze(-1)               # [M, 1]
            m = beta * torch.cholesky_solve(rhs, L).squeeze(-1)  # [M]
        except RuntimeError:
            # Fallback to pseudo-inverse if Cholesky fails
            sign, log_det_A = torch.linalg.slogdet(A)
            m = beta * torch.linalg.solve(A, Phi.T @ t)  # [M]

        # Compute the marginal log likelihood
        data_fit = 0.5 * beta * torch.sum((t - Phi @ m) ** 2)
        weight_fit = 0.5 * alpha * torch.sum(m ** 2)

        mll = (
            0.5 * M * log_alpha +
            0.5 * N * log_beta -
            0.5 * N * math.log(2 * math.pi) -
            data_fit -
            weight_fit -
            0.5 * log_det_A
        )
        
        # Add prior over hyperparameters
        if self.prior is not None:
            mll += self.prior.lnprob(theta)

        return mll

    def negative_mll(self, theta):
        """Wrapper of minimization optimizers."""
        return -self.marginal_log_likelihood(theta)
    
    def train(self, X, y, do_optimize=True):
        """
        Trains the Bayesian Linear Regression model by optimizing hyperparameters.
        Args:
            X (torch.Tensor): Input data tensor of shape (N, D) or (N, M) if basis_func is None
            y (torch.Tensor): Target values tensor of shape (N,)
            do_optimize (bool): If True, optimizes the marginal log likelihood; otherwise take initial hyperparameters.
        """
        X = X.to(self.device)
        y = y.to(self.device)
        y = y.view(-1)
        assert X.ndim == 2 and y.ndim == 1, "X must be (N, D) and y must be (N,)"
        assert X.shape[0] == y.shape[0], "Number of inputs and targets must match."

        # Save input data
        self.X, self.y = X, y

        # Build the design matrix using the chosen basis function
        if self.basis_func is None:
            self.Phi = X.to(self.device)  # [N, M]
        else:
            self.Phi = self.basis_func(X).to(self.device)  # [N, M]
        
        # Fit hyperparameters by maximizing the marginal log likelihood
        if do_optimize:
            # Optimize in log space
            theta = torch.tensor(
                [math.log(self.alpha0), math.log(self.beta0)],
                dtype=self.Phi.dtype,
                device=self.device,
                requires_grad=True,
            )

            optimizer = torch.optim.Adam([theta], lr=0.025)

            for _ in range(500):
                optimizer.zero_grad()
                mll = self.negative_mll(theta)
                mll.backward()
                optimizer.step()

            # Detach and store the optimized hyperparameters
            theta_opt = theta.detach()
        else:
            # Keep the initial hyperparameters
            theta_opt = torch.tensor(
                [math.log(self.alpha0), math.log(self.beta0)],
                dtype=self.Phi.dtype,
                device=self.device,
            )

        # Store the final hyperparameters on linear scale
        self.log_alpha, self.log_beta = theta_opt
        self.alpha = torch.exp(self.log_alpha)
        self.beta = torch.exp(self.log_beta)

    def predict(self, X_test):
        """
        Returns the predictive mean and variance of the objective function at the given test points.
        Args:
            X_test (torch.Tensor): N test data points with shape (N*, D) or (N*, M) if basis_func is None
        Returns:
            mu (torch.Tensor): Predictive mean of the test data points
            var (torch.Tensor): Predictive variance of the test data points
        """
        X_test = X_test.to(self.device)

        # Compute basis function features for train and test data
        if self.basis_func is None:
            Phi_test = X_test.to(self.device)                   # [N*, M]
        else:
            Phi_test = self.basis_func(X_test).to(self.device)  # [N*, M]
        Phi_train = self.Phi                                    # [N, M]
        t = self.y                                              # [N]

        # Posterior over weights
        alpha = self.alpha
        beta = self.beta

        M = Phi_train.shape[1]  # Number of basis functions
        I_M = torch.eye(M, device=self.device, dtype=Phi_train.dtype)
        A = alpha * I_M + beta * (Phi_train.T @ Phi_train)  # [M, M]

        # Cholesky factorization for stability
        L = torch.linalg.cholesky(A)
        S = torch.cholesky_inverse(L)                       # posterior covariance [M, M]
        rhs = (Phi_train.T @ t).unsqueeze(-1)                # [M, 1]
        m = beta * torch.cholesky_solve(rhs, L).squeeze(-1)  # posterior mean [M]

        # Predictive mean and variance
        mu = Phi_test @ m                                   # Predictive mean [N*,]
        var_model = (Phi_test @ S * Phi_test).sum(dim=1)
        var = 1. / beta + var_model

        return mu, var