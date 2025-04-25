"""
Sparse Gaussian Process Model for Bayesian Optimization.
This model is based on the GPyTorch library and mirrors the functionality
of the gpflow SGPR model, which can be found at
https://gpflow.github.io/GPflow/develop/api/gpflow/models/sgpr/https://gpflow.github.io/GPflow/develop/api/gpflow/models/sgpr/
"""

from gpytorch.models import ApproximateGP
from gpytorch.variational import VariationalStrategy, DeltaVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel


class SparseGPModel(ApproximateGP):
    """
    Sparse Gaussian Process Model for Bayesian Optimization.
    """
    def __init__(self, inducing_points, ard_dims):
        """
        Args:
            inducing_points (Tensor): Inducing points for the sparse GP.
            ard_dims (int): Number of dimensions for the ARD kernel.
        """
        # DeltaVariationalDistribution ⇒ deterministic inducing values (FITC-like)
        variational_distribution = DeltaVariationalDistribution(
            inducing_points.size(0)
        )
        # VariationalStrategy ⇒ variational distribution over inducing points
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)

        # Mean and covariance modules
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            RBFKernel(ard_num_dims=ard_dims)
        )

    def forward(self, x):
        """
        Forward pass through the model.
        Args:
            x (Tensor): Input data.
        Returns:
            MultivariateNormal: Posterior distribution at the input points.
        """
        # Compute the posterior distribution at the training points
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    @property
    def Z(self):
        """
        Convenience property to access the inducing points.
        Returns:
            Tensor: Inducing points.
        """
        return self.variational_strategy.inducing_points.detach()
    
    def predict(self, x):
        """
        Convenience method to predict the mean and variance of the posterior distribution-
        Returns the mean and variance directly unlike the forward method.
        Args:
            x (Tensor): Input data.
        Returns:
            Tensor: Mean of the posterior distribution.
            Tensor: Variance of the posterior distribution.
        """
        mult_dist = self(x)
        return mult_dist.mean, mult_dist.variance