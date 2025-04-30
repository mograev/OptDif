"""
Loss functions for various models including VAE and VQ-VAE.
Sources:
- https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/losses/contperceptual.py
- https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/losses/vqperceptual.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from taming.modules.discriminator.model import NLayerDiscriminator, weights_init
from taming.modules.losses.lpips import LPIPS
from taming.modules.losses.vqperceptual import adopt_weight, hinge_d_loss, vanilla_d_loss


def hinge_d_loss_with_r1(logits_real, logits_fake, real_imgs, gamma=5.0):
    loss = torch.mean(F.relu(1. - logits_real)) + torch.mean(F.relu(1. + logits_fake))
    grad_real = torch.autograd.grad(
        outputs=logits_real.sum(), inputs=real_imgs,
        create_graph=True
    )[0]
    r1 = 0.5 * gamma * grad_real.view(grad_real.size(0), -1).pow(2).sum(1).mean()
    return loss + r1


class LPIPSWithDiscriminator(nn.Module):
    def __init__(
        self,
        disc_start=0,
        logvar_init=0.0,
        rec_img_weight=1.0,
        rec_lat_weight=1.0,
        perceptual_weight=1.0,
        kl_weight=1e-6,
        disc_weight=1.0,
        disc_num_layers=3,
        disc_in_channels=3,
        use_actnorm=False,
        disc_loss="hinge"
    ):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla", "hinge_r1"]

        # Store loss weights
        self.rec_img_weight = rec_img_weight
        self.rec_lat_weight = rec_lat_weight
        self.perceptual_weight = perceptual_weight
        self.kl_weight = kl_weight
        self.disc_weight = disc_weight

        # Perceptual loss
        self.perceptual_loss = LPIPS().eval()

        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

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

    def calculate_adaptive_weight(self, nll_loss, gen_loss, last_layer):
        # If last_layer is a module, extract its first parameter (typically the weight)
        if isinstance(last_layer, torch.nn.Module):
            last_layer = next(iter(last_layer.parameters()))

        # Compute gradients
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        gen_grads = torch.autograd.grad(gen_loss, last_layer, retain_graph=True)[0]

        # Calculate adaptive weight
        disc_a_weight = torch.norm(nll_grads) / (torch.norm(gen_grads) + 1e-4)
        disc_a_weight = torch.clamp(disc_a_weight, 0.0, 1e4).detach()
        disc_a_weight *= self.disc_weight
        return disc_a_weight

    def forward(
        self,
        inputs,        # SD latents (real)
        recons,        # SD latents (fake)
        img_inputs,    # decoded images (real)
        img_recons,    # decoded images (fake)
        posterior,
        optimizer_idx,
        global_step,
        last_layer=None,
        split="train",
    ):
        # Reconstruction losses
        rec_lat = F.mse_loss(recons.contiguous(), inputs.contiguous(), reduction="mean")
        rec_img = F.mse_loss(img_recons.contiguous(), img_inputs.contiguous(), reduction="mean")

        # Perceptual loss
        perc_img = self.perceptual_loss(img_inputs.contiguous(), img_recons.contiguous()).mean()

        # Total NLL loss
        nll_loss = (self.rec_lat_weight * rec_lat + self.rec_img_weight * rec_img
                    + self.perceptual_weight * perc_img) / torch.exp(self.logvar) + self.logvar

        # KL loss
        kl_loss = posterior.kl().mean()

        # GAN loss
        disc_active = int(global_step >= self.discriminator_iter_start)

        if optimizer_idx == 0:
            # ---- Generator update -----------------------------------------
            logits_fake = self.discriminator(recons.contiguous())
            gen_loss = -torch.mean(logits_fake)

            if self.training:
                disc_a_weight = self.calculate_adaptive_weight(nll_loss, gen_loss, last_layer)
            else:
                disc_a_weight = torch.tensor(self.disc_weight, device=nll_loss.device)
            
            loss = nll_loss + self.kl_weight * kl_loss + disc_a_weight * disc_active * gen_loss

            log = {
                "{}/rec_lat_loss".format(split): rec_lat.detach(),
                "{}/rec_img_loss".format(split): rec_img.detach(),
                "{}/perc_img_loss".format(split): perc_img.detach(),
                "{}/logvar".format(split): self.logvar.detach(),
                "{}/nll_loss".format(split): nll_loss.detach(),
                "{}/kl_loss".format(split): kl_loss.detach(),
                "{}/gen_loss".format(split): gen_loss.detach(),
                "{}/disc_weight".format(split): disc_a_weight.detach(),
            }

        if optimizer_idx == 1:
            # ---- Discriminator update --------------------------------------

            if self.discriminator_loss is hinge_d_loss_with_r1:
                # Use R1 regularization
                with torch.enable_grad():                      # ensure grads even in eval/val
                    real_imgs = inputs.contiguous().detach()
                    real_imgs.requires_grad_(True)             # needed for R1 term
                    logits_real = self.discriminator(real_imgs)
                    logits_fake = self.discriminator(recons.contiguous().detach())
                    disc_loss = self.discriminator_loss(logits_real, logits_fake, real_imgs)
            else:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(recons.contiguous().detach())
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
    def __init__(
        self,
        disc_start,
        kl_weight=1.0,
        disc_num_layers=3,
        disc_in_channels=3,
        disc_factor=1.0,
        disc_weight=1.0,
        use_actnorm=False,
        disc_conditional=False,
        disc_loss="hinge"
    ):
        super().__init__()
        self.kl_weight = kl_weight
        self.discriminator = NLayerDiscriminator(
            input_nc=disc_in_channels,
            n_layers=disc_num_layers,
            use_actnorm=use_actnorm
        ).apply(weights_init)
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.discriminator_iter_start = disc_start
        self.disc_conditional = disc_conditional

    def forward(self, inputs, reconstructions, posterior, optimizer_idx, global_step, split="train"):
        
        if optimizer_idx == 0:
            # Generator update
            # Compute reconstruction loss
            rec_loss = torch.nn.functional.mse_loss(inputs, reconstructions, reduction="mean")
            # Compute KL divergence (clamped)
            # kl_loss = torch.mean(posterior.kl())
            kl_loss = posterior.kl()
            kl_loss = torch.clamp(kl_loss, min=0.01)
            kl_loss = torch.mean(kl_loss)
            # Compute generator loss
            logits_fake = self.discriminator(reconstructions)
            g_loss = -torch.mean(logits_fake)
            factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            # Compute total loss
            total_loss = rec_loss + self.kl_weight * kl_loss + factor * g_loss
            return total_loss, {
                "{}/rec_loss".format(split): rec_loss.detach(),
                "{}/kl_loss".format(split): kl_loss.detach(),
                "{}/g_loss".format(split): g_loss.detach(),
            }
        elif optimizer_idx == 1:
            # Discriminator update
            logits_real = self.discriminator(inputs.detach())
            logits_fake = self.discriminator(reconstructions.detach())
            factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = factor * self.disc_loss(logits_real, logits_fake)
            return d_loss, {
                "{}/disc_loss".format(split): d_loss.detach(),
            }
        else:
            raise ValueError("Invalid optimizer index: {}".format(optimizer_idx))



class SimpleVAELoss(nn.Module):
    def __init__(self, beta=0.01):
        """
        Simple VAE loss combining reconstruction loss and KL divergence.
        Args:
            beta: Weight for the KL divergence term (beta-VAE parameter).
        """
        super().__init__()
        self.beta = beta
        
    def forward(self, inputs, reconstructions, posterior, split="train"):
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
        total_loss = rec_loss + self.beta * kl_loss
        
        return total_loss, {
            "{}/rec_loss".format(split): rec_loss.detach(),
            "{}/kl_loss".format(split): kl_loss.detach(),
        }
    

class SimpleAutoencoderLoss(nn.Module):
    def __init__(self):
        """
        Simple Autoencoder loss using reconstruction loss.
        """
        super().__init__()
        
    def forward(self, inputs, reconstructions):
        """
        Compute the VAE loss.
        Args:
            inputs: Original input tensor
            reconstructions: Reconstructed output tensor
        """
        # Reconstruction loss (MSE)
        rec_loss = torch.nn.functional.mse_loss(reconstructions, inputs)
        
        # Total loss
        total_loss = rec_loss
        
        return total_loss, {
            'rec_loss': rec_loss
        }


def hinge_d_loss_with_exemplar_weights(logits_real, logits_fake, weights):
    assert weights.shape[0] == logits_real.shape[0] == logits_fake.shape[0]
    loss_real = torch.mean(F.relu(1. - logits_real), dim=[1,2,3])
    loss_fake = torch.mean(F.relu(1. + logits_fake), dim=[1,2,3])
    loss_real = (weights * loss_real).sum() / weights.sum()
    loss_fake = (weights * loss_fake).sum() / weights.sum()
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def measure_perplexity(predicted_indices, n_embed):
    # src: https://github.com/karpathy/deep-vector-quantization/blob/main/model.py
    # eval cluster perplexity. when perplexity == num_embeddings then all clusters are used exactly equally
    encodings = F.one_hot(predicted_indices, n_embed).float().reshape(-1, n_embed)
    avg_probs = encodings.mean(0)
    perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()
    cluster_use = torch.sum(avg_probs > 0)
    return perplexity, cluster_use

def l1(x, y):
    return torch.abs(x-y)


def l2(x, y):
    return torch.pow((x-y), 2)


class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_ndf=64, disc_loss="hinge", n_classes=None, perceptual_loss="lpips",
                 pixel_loss="l1"):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        assert perceptual_loss in ["lpips", "clips", "dists"]
        assert pixel_loss in ["l1", "l2"]
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        if perceptual_loss == "lpips":
            print(f"{self.__class__.__name__}: Running with LPIPS.")
            self.perceptual_loss = LPIPS().eval()
        else:
            raise ValueError(f"Unknown perceptual loss: >> {perceptual_loss} <<")
        self.perceptual_weight = perceptual_weight

        if pixel_loss == "l1":
            self.pixel_loss = l1
        else:
            self.pixel_loss = l2

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm,
                                                 ndf=disc_ndf
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        self.n_classes = n_classes

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train", predicted_indices=None):
        if codebook_loss is not None:
            codebook_loss = torch.tensor([0.]).to(inputs.device)
        #rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        rec_loss = self.pixel_loss(inputs.contiguous(), reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.tensor([0.0])

        nll_loss = rec_loss
        #nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        nll_loss = torch.mean(nll_loss)

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            try:
                d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = nll_loss + d_weight * disc_factor * g_loss + self.codebook_weight * codebook_loss.mean()

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/quant_loss".format(split): codebook_loss.detach().mean(),
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/p_loss".format(split): p_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            if predicted_indices is not None:
                assert self.n_classes is not None
                with torch.no_grad():
                    perplexity, cluster_usage = measure_perplexity(predicted_indices, self.n_classes)
                log[f"{split}/perplexity"] = perplexity
                log[f"{split}/cluster_usage"] = cluster_usage
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log


class SimpleVQVAELoss(nn.Module):
    def __init__(self):
        """
        Simple VQVAE loss combining reconstruction loss and VQ loss.
        """
        super().__init__()

    def forward(self, inputs, reconstructions, vq_loss):
        """
        Compute the VQVAE loss.
        Args:
            inputs: Original input tensor
            reconstructions: Reconstructed output tensor
            vq_loss: Vector Quantization loss
        """
        # Reconstruction loss (MSE)
        rec_loss = torch.nn.functional.mse_loss(reconstructions, inputs)

        # Total loss
        total_loss = rec_loss + vq_loss

        return total_loss, {
            'rec_loss': rec_loss,
            'vq_loss': vq_loss
        }