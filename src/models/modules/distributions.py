"""
Distributions used in Stable Diffusion models.
Source: https://github.com/Stability-AI/stablediffusion/blob/main/ldm/modules/distributions/distributions.py
"""

import torch
import numpy as np


class DiagonalGaussianDistribution(object):
    """
    Diagonal Gaussian distribution with mean and variance.
    This class is used to represent a Gaussian distribution with diagonal covariance.
    It is used in the VAE model to represent the posterior distribution of the latent variables.
    """
    def __init__(self, parameters, deterministic=False):
        """
        Args:
            parameters (torch.Tensor): The parameters of the distribution.
                It should be a tensor of shape (B, 2 * C, H, W) for 2D images or
                (B, 2 * L) for flattened latents.
            deterministic (bool): If True, the distribution is deterministic.
        """
        self.parameters = parameters
        # Check whether the parameters are 2D or 4D
        self.dims = [1, 2, 3] if len(parameters.shape) == 4 else [1]
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        """
        Sample from the distribution.
        Returns:
            torch.Tensor: A sample from the distribution.
        """
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        """
        Compute the KL divergence between two distributions.
        Args:
            other (DiagonalGaussianDistribution): The other distribution to compute the KL divergence with.
                If None, compute the KL divergence of the distribution with itself.
        Returns:
            torch.Tensor: The KL divergence.
        """
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=self.dims)
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=self.dims)

    def nll(self, sample, dims=None):
        """
        Compute the negative log likelihood of a sample.
        Args:
            sample (torch.Tensor): The sample to compute the negative log likelihood for.
            dims (list): The dimensions to sum over.
        Returns:
            torch.Tensor: The negative log likelihood.
        """
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims if dims is not None else self.dims)

    def mode(self):
        """Compute the mode of the distribution."""
        return self.mean
    
    def detach(self):
        """Detach the distribution from the computation graph."""
        return DiagonalGaussianDistribution(self.parameters.detach(), self.deterministic)


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians. Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    Source: https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/losses.py#L12
    Args:
        mean1 (torch.Tensor): The mean of the first distribution.
        logvar1 (torch.Tensor): The log variance of the first distribution.
        mean2 (torch.Tensor): The mean of the second distribution.
        logvar2 (torch.Tensor): The log variance of the second distribution.
    Returns:
        torch.Tensor: The KL divergence between the two distributions.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for torch.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )