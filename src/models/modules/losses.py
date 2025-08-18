"""
Loss functions for various models including VAE and VQ-VAE.
Based on the implementations from the original Stable Diffusion repository,
but adapted to latent model training with various model families.
Sources:
- https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/losses/contperceptual.py
- https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/losses/vqperceptual.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from taming.modules.discriminator.model import NLayerDiscriminator, weights_init
from taming.modules.losses.lpips import LPIPS
from taming.modules.losses.vqperceptual import hinge_d_loss, vanilla_d_loss


def hinge_d_loss_with_r1(logits_real, logits_fake, real_imgs, gamma=5.0):
    """
    Hinge loss with R1 regularization.
    Args:
        logits_real: Discriminator output for real images.
        logits_fake: Discriminator output for fake images.
        real_imgs: Real images used for R1 regularization.
        gamma: Weight for the R1 regularization term.
    Returns:
        loss: Total loss including hinge loss and R1 regularization.
    """
    loss = torch.mean(F.relu(1. - logits_real)) + torch.mean(F.relu(1. + logits_fake))
    grad_real = torch.autograd.grad(
        outputs=logits_real.sum(), inputs=real_imgs,
        create_graph=True
    )[0]
    r1 = 0.5 * gamma * grad_real.view(grad_real.size(0), -1).pow(2).sum(1).mean()
    return loss + r1

def measure_perplexity(predicted_indices, n_classes):
    """
    Measure the perplexity of the predicted indices.
    Args:
        predicted_indices: Predicted indices for clustering.
        n_classes: Number of classes for clustering.
    Returns:
        perplexity: Perplexity value.
        cluster_usage: Cluster usage statistics.
    """
    # Calculate the number of samples in each cluster
    cluster_usage = torch.bincount(predicted_indices.reshape(-1), minlength=n_classes).float()

    # Calculate the perplexity
    p = cluster_usage / cluster_usage.sum()
    perplexity = torch.exp(-torch.sum(p * torch.log(p + 1e-10)))

    return perplexity, cluster_usage


class LPIPSWithDiscriminator(nn.Module):
    """
    LPIPS + MSE + KL + Discriminator loss for variational autoencoder.
    """
    def __init__(self,
                 disc_start=0,
                 rec_img_weight=1.0,
                 rec_lat_weight=1.0,
                 perceptual_weight=1.0,
                 kl_weight=1e-6,
                 disc_weight=1.0,
                 disc_num_layers=3,
                 disc_in_channels=3,
                 use_actnorm=False,
                 disc_loss="hinge",
                 pixel_loss="l2"
                ):
        """
        LPIPS + MSE loss with discriminator for linear autoencoder.
        Args:
            disc_start: Step at which the discriminator starts training.
            rec_img_weight: Weight for the image reconstruction loss term.
            rec_lat_weight: Weight for the latent reconstruction loss term.
            perceptual_weight: Weight for the perceptual loss term.
            kl_weight: Weight for the KL divergence term.
            disc_weight: Weight for the discriminator loss term.
            disc_num_layers: Number of layers in the discriminator.
            disc_in_channels: Number of input channels for the discriminator.
            use_actnorm: Whether to use activation normalization in the discriminator.
            disc_loss: Type of discriminator loss function to use ("hinge", "vanilla", "hinge_r1").
        """
        super().__init__()
        assert disc_loss in ["hinge", "vanilla", "hinge_r1"]

        # Store loss weights
        self.rec_img_weight = rec_img_weight
        self.rec_lat_weight = rec_lat_weight
        self.perceptual_weight = perceptual_weight
        self.kl_weight = kl_weight
        self.disc_weight = disc_weight

        # Reconstruction loss
        if pixel_loss == "l1":
            self.pixel_loss = torch.nn.L1Loss(reduction="mean")
        elif pixel_loss == "l2":
            self.pixel_loss = torch.nn.MSELoss(reduction="mean")

        # Perceptual loss
        self.perceptual_loss = LPIPS().eval()

        # Discriminator
        self.discriminator = NLayerDiscriminator(
            input_nc=disc_in_channels,
            n_layers=disc_num_layers,
            use_actnorm=use_actnorm
        ).apply(weights_init)
        self.discriminator_iter_start = disc_start

        # Discriminator loss
        if disc_loss == "hinge":
            self.discriminator_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.discriminator_loss = vanilla_d_loss
        elif disc_loss == "hinge_r1":
            # Use R1 regularization
            self.discriminator_loss = hinge_d_loss_with_r1

    def forward(self, inputs, recons, img_inputs, img_recons, posterior, optimizer_idx, global_step, split="train"):
        """
        Forward pass for the LPIPS + MSE loss with discriminator.
        Args:
            inputs: Original input tensor (SD latents).
            recons: Reconstructed output tensor (SD latents).
            img_inputs: Original image tensor (decoded images).
            img_recons: Reconstructed image tensor (decoded images).
            posterior: Posterior distribution.
            optimizer_idx: Index of the optimizer (0 for generator, 1 for discriminator).
            global_step: Current training step.
            split: Split name (train/val/test) for logging purposes.
        """
        # Whether to use discriminator
        disc_active = int(global_step >= self.discriminator_iter_start)

        if optimizer_idx == 0:
            # ---- Generator update ------------------------------- #

            # Reconstruction losses
            rec_lat = self.pixel_loss(recons.contiguous(), inputs.contiguous())
            rec_img = self.pixel_loss(img_recons.contiguous(), img_inputs.contiguous())

            # Perceptual loss
            perc_img = self.perceptual_loss(img_inputs.contiguous(), img_recons.contiguous()).mean()

            # Total NLL loss
            nll_loss = (self.rec_lat_weight * rec_lat + self.rec_img_weight * rec_img
                        + self.perceptual_weight * perc_img)

            # KL loss
            kl_loss = posterior.kl().mean()

            # Generator loss
            logits_fake = self.discriminator(img_recons.contiguous())
            gen_loss = -torch.mean(logits_fake)

            loss = nll_loss + self.kl_weight * kl_loss + self.disc_weight * disc_active * gen_loss

            log = {
                "{}/rec_lat_loss".format(split): rec_lat.detach(),
                "{}/rec_img_loss".format(split): rec_img.detach(),
                "{}/perc_img_loss".format(split): perc_img.detach(),
                "{}/nll_loss".format(split): nll_loss.detach(),
                "{}/kl_loss".format(split): kl_loss.detach(),
                "{}/gen_loss".format(split): gen_loss.detach(),
            }

        if optimizer_idx == 1:
            # ---- Discriminator update --------------------------- #

            if self.discriminator_loss is hinge_d_loss_with_r1:
                # Use R1 regularization
                with torch.enable_grad():                      # ensure grads even in eval/val
                    real_imgs = img_inputs.contiguous().detach()
                    real_imgs.requires_grad_(True)             # needed for R1 term
                    logits_real = self.discriminator(real_imgs)
                    logits_fake = self.discriminator(img_recons.contiguous().detach())
                    disc_loss = self.discriminator_loss(logits_real, logits_fake, real_imgs)
            else:
                logits_real = self.discriminator(img_inputs.contiguous().detach())
                logits_fake = self.discriminator(img_recons.contiguous().detach())
                disc_loss = self.discriminator_loss(logits_real, logits_fake)

            loss = disc_active * disc_loss

            log = {
                "{}/disc_loss".format(split): disc_loss.detach(),
                "{}/disc_active".format(split): disc_active,
                "{}/logits_real".format(split): logits_real.detach().mean(),
                "{}/logits_fake".format(split): logits_fake.detach().mean()
            }

        return loss, log


class VAEWithDiscriminator(nn.Module):
    """
    MSE + KL + Discriminator loss for variational autoencoder.
    """
    def __init__(self,
                 disc_start=0,
                 kl_weight=1.0,
                 disc_weight=1.0,
                 disc_num_layers=3,
                 disc_in_channels=3,
                 use_actnorm=False,
                 disc_loss="hinge"
                ):
        """
        VAE with discriminator loss.
        Args:
            disc_start: Step at which the discriminator starts training.
            kl_weight: Weight for the KL divergence term.
            disc_weight: Weight for the discriminator loss term.
            disc_num_layers: Number of layers in the discriminator.
            disc_in_channels: Number of input channels for the discriminator.
            use_actnorm: Whether to use activation normalization in the discriminator.
            disc_loss: Type of discriminator loss function to use ("hinge", "vanilla", "hinge_r1").
        """
        super().__init__()

        # Loss weights
        self.kl_weight = kl_weight
        self.disc_weight = disc_weight

        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * 0.0)

        # Discriminator
        self.discriminator = NLayerDiscriminator(
            input_nc=disc_in_channels,
            n_layers=disc_num_layers,
            use_actnorm=use_actnorm
        ).apply(weights_init)
        self.discriminator_iter_start = disc_start

        # Discriminator loss
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        elif disc_loss == "hinge_r1":
            # Use R1 regularization
            self.disc_loss = hinge_d_loss_with_r1

    def forward(self, inputs, recons, posterior, optimizer_idx, global_step, split="train"):
        """
        Forward pass for the VAE with discriminator. Here the discriminator works in the latent space.
        Args:
            inputs: Original input tensor.
            recons: Reconstructed output tensor.
            posterior: Posterior distribution.
            optimizer_idx: Index of the optimizer (0 for generator, 1 for discriminator).
            global_step: Current training step.
            last_layer: Last layer of the model (for gradient calculation).
            split: Split name (train/val/test) for logging purposes.
        """
        # Whether to use discriminator
        disc_active = int(global_step >= self.discriminator_iter_start)

        if optimizer_idx == 0:
            # ---- Generator update ------------------------------- #

            # Reconstruction loss
            rec_loss = F.mse_loss(recons.contiguous(), inputs.contiguous(), reduction="mean")

            # KL loss
            kl_loss = posterior.kl().mean()

            # Generator loss
            logits_fake = self.discriminator(recons.contiguous())
            gen_loss = -torch.mean(logits_fake)

            loss = rec_loss + self.kl_weight * kl_loss + self.disc_weight * disc_active * gen_loss

            log = {
                "{}/rec_loss".format(split): rec_loss.detach(),
                "{}/kl_loss".format(split): kl_loss.detach(),
                "{}/gen_loss".format(split): gen_loss.detach(),
            }

        elif optimizer_idx == 1:
            # ---- Discriminator update --------------------------- #

            logits_real = self.discriminator(inputs.contiguous().detach())
            logits_fake = self.discriminator(recons.contiguous().detach())
            disc_loss = self.disc_loss(logits_real, logits_fake)

            loss = disc_active * disc_loss

            log = {
                "{}/disc_loss".format(split): disc_loss.detach(),
                "{}/disc_active".format(split): disc_active,
                "{}/logits_real".format(split): logits_real.detach().mean(),
                "{}/logits_fake".format(split): logits_fake.detach().mean()
            }

        return loss, log


class SimpleVAELoss(nn.Module):
    """
    Simple VAE loss combining reconstruction loss and KL divergence.
    """
    def __init__(self, beta=0.01):
        """
        Simple VAE loss combining reconstruction loss and KL divergence.
        Args:
            beta: Weight for the KL divergence term (beta-VAE parameter).
        """
        super().__init__()
        self.beta = beta

    def forward(self, inputs, recons, posterior, split="train"):
        """
        Compute the VAE loss.
        Args:
            inputs: Original input tensor
            recons: Reconstructed output tensor
            posterior: DiagonalGaussianDistribution instance
            split: Split name (train/val/test) for logging purposes
        """
        # Reconstruction loss (MSE)
        rec_loss = torch.nn.functional.mse_loss(recons, inputs)

        # KL divergence
        kl_loss = posterior.kl().mean()

        # Total loss
        total_loss = rec_loss + self.beta * kl_loss

        return total_loss, {
            "{}/rec_loss".format(split): rec_loss.detach(),
            "{}/kl_loss".format(split): kl_loss.detach(),
        }


class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(self,
                 disc_start=0,
                 rec_img_weight=1.0,
                 rec_lat_weight=1.0,
                 perceptual_weight=1.0,
                 codebook_weight=1.0,
                 disc_weight=1.0,
                 disc_num_layers=3,
                 disc_in_channels=3,
                 use_actnorm=False,
                 disc_loss="hinge",
                 pixel_loss="l1",
                 n_embed=None,
                ):
        """
        VQ + LPIPS + Discriminator loss for VQ-VAE.
        Args:
            disc_start: Step at which the discriminator starts training.
            rec_img_weight: Weight for the image reconstruction loss term.
            rec_lat_weight: Weight for the latent reconstruction loss term.
            perceptual_weight: Weight for the perceptual loss term.
            codebook_weight: Weight for the codebook loss term.
            disc_weight: Weight for the discriminator loss term.
            disc_num_layers: Number of layers in the discriminator.
            disc_in_channels: Number of input channels for the discriminator.
            use_actnorm: Whether to use activation normalization in the discriminator.
            disc_loss: Type of discriminator loss function to use ("hinge", "vanilla").
            pixel_loss: Type of pixel loss function to use ("l1", "l2").
        """
        super().__init__()
        assert disc_loss in ["hinge", "vanilla", "hinge_r1"]
        assert pixel_loss in ["l1", "l2"]

        # Store loss weights
        self.rec_img_weight = rec_img_weight
        self.rec_lat_weight = rec_lat_weight
        self.perceptual_weight = perceptual_weight
        self.codebook_weight = codebook_weight
        self.disc_weight = disc_weight

        # Reconstruction loss
        if pixel_loss == "l1":
            self.pixel_loss = torch.nn.L1Loss(reduction="mean")
        elif pixel_loss == "l2":
            self.pixel_loss = torch.nn.MSELoss(reduction="mean")

        # Perceptual loss
        self.perceptual_loss = LPIPS().eval()

        # Discriminator
        self.discriminator = NLayerDiscriminator(
            input_nc=disc_in_channels,
            n_layers=disc_num_layers,
            use_actnorm=use_actnorm,
        ).apply(weights_init)
        self.discriminator_iter_start = disc_start

        # Discriminator loss
        if disc_loss == "hinge":
            self.discriminator_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.discriminator_loss = vanilla_d_loss
        elif disc_loss == "hinge_r1":
            # Use R1 regularization
            self.discriminator_loss = hinge_d_loss_with_r1

        # Number of classes for clustering
        # This is used for perplexity calculation
        self.n_classes = n_embed

    def forward(self, inputs, recons, img_inputs, img_recons, codebook_loss, optimizer_idx,
                global_step, predicted_indices=None, split="train"):
        """
        Forward pass for the VQ + LPIPS + Discriminator loss.
        Args:
            inputs: Original input tensor (SD latents).
            recons: Reconstructed output tensor (SD latents).
            img_inputs: Original image tensor (decoded images).
            img_recons: Reconstructed image tensor (decoded images).
            codebook_loss: Codebook loss.
            optimizer_idx: Index of the optimizer (0 for generator, 1 for discriminator).
            global_step: Current training step.
            predicted_indices: Predicted indices for clustering.
            split: Split name (train/val/test) for logging purposes.
        """
        # Whether to use discriminator
        disc_active = int(global_step >= self.discriminator_iter_start)

        if optimizer_idx == 0:
            # ---- Generator update ------------------------------- #

            # Reconstruction loss
            rec_lat = self.pixel_loss(recons.contiguous(), inputs.contiguous())
            rec_img = self.pixel_loss(img_recons.contiguous(), img_inputs.contiguous())

            # Perceptual loss
            perc_img = self.perceptual_loss(img_inputs.contiguous(), img_recons.contiguous()).mean()

            # Total NLL loss
            nll_loss = self.rec_lat_weight * rec_lat + self.rec_img_weight * rec_img + self.perceptual_weight * perc_img

            # Check codebook loss
            if codebook_loss is None:
                codebook_loss = torch.tensor([0.]).to(inputs.device)

            # Generator loss
            logits_fake = self.discriminator(img_recons.contiguous())
            gen_loss = -torch.mean(logits_fake)

            loss = nll_loss + self.codebook_weight * codebook_loss + self.disc_weight * disc_active * gen_loss

            log = {
                "{}/rec_lat_loss".format(split): rec_lat.detach(),
                "{}/rec_img_loss".format(split): rec_img.detach(),
                "{}/perc_img_loss".format(split): perc_img.detach(),
                "{}/codebook_loss".format(split): codebook_loss.detach(),
                "{}/nll_loss".format(split): nll_loss.detach(),
                "{}/gen_loss".format(split): gen_loss.detach(),
            }

            # Add perplexity to the log
            if predicted_indices is not None:
                assert self.n_classes is not None
                with torch.no_grad():
                    perplexity, _ = measure_perplexity(predicted_indices, self.n_classes)
                log[f"{split}/perplexity"] = perplexity

        if optimizer_idx == 1:
            # ---- Discriminator update --------------------------- #

            if self.discriminator_loss is hinge_d_loss_with_r1:
                # Use R1 regularization
                with torch.enable_grad():                      # ensure grads even in eval/val
                    real_imgs = img_inputs.contiguous().detach()
                    real_imgs.requires_grad_(True)             # needed for R1 term
                    logits_real = self.discriminator(real_imgs)
                    logits_fake = self.discriminator(img_recons.contiguous().detach())
                    disc_loss = self.discriminator_loss(logits_real, logits_fake, real_imgs)
            else:
                logits_real = self.discriminator(img_inputs.contiguous().detach())
                logits_fake = self.discriminator(img_recons.contiguous().detach())
                disc_loss = self.discriminator_loss(logits_real, logits_fake)

            loss = disc_active * disc_loss

            log = {
                "{}/disc_loss".format(split): disc_loss.detach(),
                "{}/disc_active".format(split): disc_active,
                "{}/logits_real".format(split): logits_real.detach().mean(),
                "{}/logits_fake".format(split): logits_fake.detach().mean()
            }

        return loss, log


class SimpleVQVAELoss(nn.Module):
    """
    Simple VQVAE loss combining reconstruction loss and VQ loss.
    """
    def forward(self, inputs, recons, vq_loss, split="train"):
        """
        Compute the VQVAE loss.
        Args:
            inputs: Original input tensor
            recons: Reconstructed output tensor
            vq_loss: Vector Quantization loss
            split: Split name (train/val/test) for logging purposes
        """
        # Reconstruction loss (MSE)
        rec_loss = torch.nn.functional.mse_loss(recons, inputs)

        # Total loss
        total_loss = rec_loss + vq_loss

        return total_loss, {
            "{}/rec_loss".format(split): rec_loss.detach(),
            "{}/vq_loss".format(split): vq_loss.detach(),
        }


class AutoencoderLPIPSWithDiscriminator(nn.Module):
    """
    Reconstruction + LPIPS + Discriminator loss for linear autoencoder.
    """
    def __init__(self,
                 disc_start=0,
                 rec_img_weight=1.0,
                 rec_lat_weight=1.0,
                 perceptual_weight=1.0,
                 disc_weight=1.0,
                 disc_num_layers=3,
                 disc_in_channels=3,
                 use_actnorm=False,
                 disc_loss="hinge",
                 pixel_loss="l1",
                ):
        """
        Reconstruction + LPIPS + Discriminator loss for linear autoencoder.
        Args:
            disc_start: Step at which the discriminator starts training.
            rec_img_weight: Weight for the image reconstruction loss term.
            rec_lat_weight: Weight for the latent reconstruction loss term.
            perceptual_weight: Weight for the perceptual loss term.
            disc_weight: Weight for the discriminator loss term.
            disc_num_layers: Number of layers in the discriminator.
            disc_in_channels: Number of input channels for the discriminator.
            use_actnorm: Whether to use activation normalization in the discriminator.
            disc_loss: Type of discriminator loss function to use ("hinge", "vanilla", "hinge_r1").
            pixel_loss: Type of pixel loss function to use ("l1", "l2").
        """
        super().__init__()
        assert disc_loss in ["hinge", "vanilla", "hinge_r1"]
        assert pixel_loss in ["l1", "l2"]

        # Store loss weights
        self.rec_img_weight = rec_img_weight
        self.rec_lat_weight = rec_lat_weight
        self.perceptual_weight = perceptual_weight

        # Reconstruction loss
        if pixel_loss == "l1":
            self.pixel_loss = torch.nn.L1Loss(reduction="mean")
        elif pixel_loss == "l2":
            self.pixel_loss = torch.nn.MSELoss(reduction="mean")
        self.disc_weight = disc_weight

        # Perceptual loss
        self.perceptual_loss = LPIPS().eval()

        # Discriminator
        self.discriminator = NLayerDiscriminator(
            input_nc=disc_in_channels,
            n_layers=disc_num_layers,
            use_actnorm=use_actnorm
        ).apply(weights_init)
        self.discriminator_iter_start = disc_start

        # Discriminator loss
        if disc_loss == "hinge":
            self.discriminator_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.discriminator_loss = vanilla_d_loss
        elif disc_loss == "hinge_r1":
            # Use R1 regularization
            self.discriminator_loss = hinge_d_loss_with_r1

    def forward(self, inputs, recons, img_inputs, img_recons, optimizer_idx, global_step, split="train"):
        """
        Compute the forward pass for the model.
        Args:
            inputs: Original input tensor
            recons: Reconstructed output tensor
            img_inputs: Original image tensor
            img_recons: Reconstructed image tensor
            optimizer_idx: Index of the optimizer
            global_step: Global step counter
            split: Split name (train/val/test) for logging purposes
        """
        # Whether to use discriminator
        disc_active = int(global_step >= self.discriminator_iter_start)

        if optimizer_idx == 0:
            # ---- Generator update ------------------------------- #

            # Reconstruction losses
            rec_lat = self.pixel_loss(recons.contiguous(), inputs.contiguous())
            rec_img = self.pixel_loss(img_recons.contiguous(), img_inputs.contiguous())

            # Perceptual loss
            perc_img = self.perceptual_loss(img_inputs.contiguous(), img_recons.contiguous()).mean()

            # Total NLL loss
            nll_loss = (self.rec_lat_weight * rec_lat + self.rec_img_weight * rec_img
                        + self.perceptual_weight * perc_img)

            # Generator loss
            logits_fake = self.discriminator(img_recons.contiguous())
            gen_loss = -torch.mean(logits_fake)

            loss = nll_loss + self.disc_weight * disc_active * gen_loss

            log = {
                "{}/rec_img_loss".format(split): rec_img.detach(),
                "{}/rec_lat_loss".format(split): rec_lat.detach(),
                "{}/perc_img_loss".format(split): perc_img.detach(),
                "{}/nll_loss".format(split): nll_loss.detach(),
                "{}/gen_loss".format(split): gen_loss.detach(),
            }

        elif optimizer_idx == 1:
            # ---- Discriminator update --------------------------- #

            if self.discriminator_loss is hinge_d_loss_with_r1:
                # Use R1 regularization
                with torch.enable_grad():                      # ensure grads even in eval/val
                    real_imgs = img_inputs.contiguous().detach()
                    real_imgs.requires_grad_(True)             # needed for R1 term
                    logits_real = self.discriminator(real_imgs)
                    logits_fake = self.discriminator(img_recons.contiguous().detach())
                    disc_loss = self.discriminator_loss(logits_real, logits_fake, real_imgs)
            else:
                logits_real = self.discriminator(img_inputs.contiguous().detach())
                logits_fake = self.discriminator(img_recons.contiguous().detach())
                disc_loss = self.discriminator_loss(logits_real, logits_fake)

            loss = disc_active * disc_loss

            log = {
                "{}/disc_loss".format(split): disc_loss.detach(),
                "{}/disc_active".format(split): disc_active,
                "{}/logits_real".format(split): logits_real.detach().mean(),
                "{}/logits_fake".format(split): logits_fake.detach().mean()
            }

        return loss, log


class SimpleAutoencoderLoss(nn.Module):
    """
    Simple Autoencoder loss using reconstruction loss.
    """
    def forward(self, inputs, reconstructions, split="train"):
        """
        Compute the VAE loss.
        Args:
            inputs: Original input tensor
            reconstructions: Reconstructed output tensor
            split: Split name (train/val/test) for logging purposes
        """
        # Reconstruction loss (MSE)
        rec_loss = torch.nn.functional.mse_loss(reconstructions, inputs)

        # Total loss
        total_loss = rec_loss

        return total_loss, {
            "{}/rec_loss".format(split): rec_loss.detach(),
        }


class LPIPSMSELoss(nn.Module):
    def __init__(self, rec_img_weight=1.0, rec_lat_weight=1.0, lpips_weight=1.0):
        """
        LPIPS + MSE loss for linear autoencoder.
        Args:
            rec_img_weight: Weight for the image reconstruction loss term.
            rec_lat_weight: Weight for the latent reconstruction loss term.
            lpips_weight: Weight for the LPIPS loss term.
        """
        super().__init__()

        # Loss weights
        self.rec_img_weight = rec_img_weight
        self.rec_lat_weight = rec_lat_weight
        self.lpips_weight = lpips_weight

        # LPIPS loss
        self.lpips_loss = LPIPS().eval()

    def forward(self, inputs, recons, inputs_img, recons_img, split="train"):
        """
        Compute the LPIPS + MSE loss.
        Args:
            inputs: Original input tensor
            recons: Reconstructed output tensor
            inputs_img: Original image tensor
            recons_img: Reconstructed image tensor
            split: Split name (train/val/test) for logging purposes
        """
        # LPIPS loss
        lpips_loss = self.lpips_loss(inputs_img.contiguous(), recons_img.contiguous()).mean()

        # MSE loss
        rec_lat_loss = torch.nn.functional.mse_loss(inputs, recons)
        rec_img_loss = torch.nn.functional.mse_loss(inputs_img, recons_img)

        # Total loss
        total_loss = (
            self.rec_lat_weight * rec_lat_loss +
            self.rec_img_weight * rec_img_loss +
            self.lpips_weight * lpips_loss
        )

        return total_loss, {
            "{}/rec_lat_loss".format(split): rec_lat_loss.detach(),
            "{}/rec_img_loss".format(split): rec_img_loss.detach(),
            "{}/lpips_loss".format(split): lpips_loss.detach(),
            "{}/total_loss".format(split): total_loss.detach()
        }


class SDVAELoss(nn.Module):
    """
    Loss for training a Stable Diffusion Variational Autoencoder.
    """
    def __init__(self,
                 rec_weight=1.0,
                 perceptual_weight=1.0,
                 kl_weight=1e-6,
                 pixel_loss="l2"
                ):
        """
        LPIPS + MSE + KL loss for Stable Diffusion VAE.
        Args:
            rec_weight: Weight for the image reconstruction loss term.
            perceptual_weight: Weight for the perceptual loss term.
            kl_weight: Weight for the KL divergence term.
            pixel_loss: Type of pixel loss function to use ("l1", "l2").
        """
        super().__init__()
        assert pixel_loss in ["l1", "l2"]

        # Store loss weights
        self.rec_weight = rec_weight
        self.perceptual_weight = perceptual_weight
        self.kl_weight = kl_weight
        self.pixel_loss = pixel_loss

        # Reconstruction loss
        if pixel_loss == "l1":
            self.pixel_loss = torch.nn.L1Loss(reduction="mean")
        elif pixel_loss == "l2":
            self.pixel_loss = torch.nn.MSELoss(reduction="mean")

        # Perceptual loss
        self.perceptual_loss = LPIPS().eval()

    def forward(self, inputs, recons, latents, split="train"):
        """
        Forward pass for loss computation.
        Args:
            inputs: Original input tensor (SD latents).
            recons: Reconstructed output tensor (SD latents).
            split: Split name (train/val/test) for logging purposes.
        """
        # Reconstruction loss
        rec_loss = self.pixel_loss(recons.contiguous(), inputs.contiguous())

        # Perceptual loss
        perc_img = self.perceptual_loss(inputs.contiguous(), recons.contiguous()).mean()

        # KL loss
        kl_loss = -0.5 * torch.sum(
            1 + latents.logvar - latents.mean.pow(2) - latents.var
        ) / inputs.shape[0]

        # Total loss
        loss = self.rec_weight * rec_loss + self.perceptual_weight * perc_img + self.kl_weight * kl_loss

        log = {
            "{}/rec_loss".format(split): rec_loss.detach(),
            "{}/perc_img_loss".format(split): perc_img.detach(),
            "{}/kl_loss".format(split): kl_loss.detach(),
            "{}/total_loss".format(split): loss.detach()
        }

        return loss, log