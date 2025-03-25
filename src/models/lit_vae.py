import argparse

import torch
import pytorch_lightning as pl

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from diffusers import AutoencoderKL


# class PlAutoencoderKL(pl.LightningModule):
#     def __init__(self, lr=1e-4, beta=1.0, pretrained_model_path=None):
#         super().__init__()
#         self.save_hyperparameters()

#         if pretrained_model_path:
#             self.vae = AutoencoderKL.from_pretrained(pretrained_model_path)
#         else:
#             self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")

#         self.lr = lr
#         self.beta = beta

#     def forward(self, x):
#         latents = self.vae.encode(x).latent_dist
#         z = latents.sample() * 0.18215  # scaling factor for SD latent space
#         recon = self.vae.decode(z).sample
#         return recon, latents

#     def training_step(self, batch, batch_idx):
#         x = batch["pixel_values"]  # assumes DataLoader returns dict with 'pixel_values'
#         recon, latents = self.forward(x)

#         recon_loss = F.mse_loss(recon, x, reduction="mean")
#         kl_loss = -0.5 * torch.sum(
#             1 + latents.logvar - latents.mean.pow(2) - latents.stddev.pow(2)
#         ) / x.shape[0]

#         loss = recon_loss + self.beta * kl_loss

#         self.log_dict({
#             "train_loss": loss,
#             "recon_loss": recon_loss,
#             "kl_loss": kl_loss
#         })

#         return loss

#     def configure_optimizers(self):
#         return torch.optim.AdamW(self.parameters(), lr=self.lr)

#     def validation_step(self, batch, batch_idx):
#         x = batch["pixel_values"]
#         recon, latents = self.forward(x)

#         recon_loss = F.mse_loss(recon, x, reduction="mean")
#         kl_loss = -0.5 * torch.sum(
#             1 + latents.logvar - latents.mean.pow(2) - latents.stddev.pow(2)
#         ) / x.shape[0]

#         loss = recon_loss + self.beta * kl_loss

#         self.log_dict({
#             "val_loss": loss,
#             "val_recon_loss": recon_loss,
#             "val_kl_loss": kl_loss
#         }, prog_bar=True)

#         return loss


class LitVAE(pl.LightningModule):

    def __init__(self, model, lr=1e-4, beta=1.0):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.beta = beta

        # store model
        self.model = model
        
        # extract hyperparameters from diffusers vae config
        self.config = model.config
        self.scaling_factor = self.config["scaling_factor"]
        self.shift_factor = self.config["shift_factor"]

    def forward(self, x):
        # 1. Encode
        latents = self.model.encode(x).latent_dist
        # 2. Sample
        z = latents.sample() * self.scaling_factor + self.shift_factor
        # 3. Decode
        recon = self.model.decode(z).sample
        return recon, latents
    
    def training_step(self, batch, batch_idx):
        # Forward pass
        x = batch["pixel_values"]
        recon, latents = self.forward(x)

        # Reconstruction loss
        recon_loss = F.mse_loss(recon, x, reduction="mean")

        # KL divergence
        kl_loss = -0.5 * torch.sum(
            1 + latents.logvar - latents.mean.pow(2) - latents.stddev.pow(2)
        ) / x.shape[0]

        # Total loss
        loss = recon_loss + self.beta * kl_loss

        # Log losses
        self.log_dict({
            "train_loss": loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss
        })

        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
    
    def validation_step(self, batch, batch_idx):
        # Forward pass
        x = batch["pixel_values"]
        recon, latents = self.forward(x)

        # Reconstruction loss
        recon_loss = F.mse_loss(recon, x, reduction="mean")

        # KL divergence
        kl_loss = -0.5 * torch.sum(
            1 + latents.logvar - latents.mean.pow(2) - latents.stddev.pow(2)
        ) / x.shape[0]

        # Total loss
        loss = recon_loss + self.beta * kl_loss

        # Log losses
        self.log_dict({
            "val_loss": loss,
            "val_recon_loss": recon_loss,
            "val_kl_loss": kl_loss
        }, prog_bar=True)

        return loss