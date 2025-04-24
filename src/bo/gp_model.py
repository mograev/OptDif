from gpytorch.models import ApproximateGP
from gpytorch.variational import VariationalStrategy, DeltaVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel


class SparseGPModel(ApproximateGP):
    def __init__(self, inducing_points, ard_dims):
        # DeltaVariationalDistribution ⇒ deterministic inducing values (FITC-like)
        variational_distribution = DeltaVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)

        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            RBFKernel(ard_num_dims=ard_dims)
        )

    def forward(self, x):
        # Compute the posterior distribution at the training points
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    @property
    def Z(self):  # convenience
        return self.variational_strategy.inducing_points.detach()
    
    def predict(self, x):
        # For compatibility with the original optimization code
        # Returns the mean and variance of the posterior distribution
        mult_dist = self(x)
        return mult_dist.mean, mult_dist.variance