import torch
import torch.nn as nn

class SimpleVAELoss(nn.Module):
    def __init__(self, beta=0.01):
        """
        Simple VAE loss combining reconstruction loss and KL divergence.
        Args:
            beta: Weight for the KL divergence term (beta-VAE parameter).
        """
        super().__init__()
        self.beta = beta
        
    def forward(self, inputs, reconstructions, posterior):
        """
        Compute the VAE loss.
        Args:
            inputs: Original input tensor
            reconstructions: Reconstructed output tensor
            posterior: DiagonalGaussianDistribution instance
        """
        # Reconstruction loss (MSE)
        rec_loss = torch.nn.functional.mse_loss(reconstructions, inputs)
        
        # KL divergence
        kl_loss = posterior.kl().mean()
        
        # Total loss
        loss = rec_loss + self.beta * kl_loss
        
        return {
            'loss': loss,
            'rec_loss': rec_loss,
            'kl_loss': kl_loss
        }