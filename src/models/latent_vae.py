import torch
import torch.nn.functional as F
import pytorch_lightning as pl

import argparse

from src.models.modules.autoencoder import Encoder, Decoder
from src.models.modules.distributions import DiagonalGaussianDistribution
from src.models.modules.utils import instantiate_from_config


class LatentVAE(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 monitor=None,
                 ):
        """
        LatentVAE is a modified version of the AutoencoderKL that processes latents instead of images.
        It uses the same encoder and decoder architecture but is designed to work with the latent space
        of a diffusion model.
        Args:
            ddconfig: Configuration for the encoder and decoder.
            lossconfig: Configuration for the loss function.
            embed_dim: The dimensionality of the flat latent space (e.g. 128).
            input_channels: Number of channels in the input latents.
            ckpt_path: Path to a checkpoint to load weights from.
            ignore_keys: Keys to ignore when loading weights from the checkpoint.
            monitor: Metric to monitor for early stopping or model selection.
        """
        super().__init__()
        self.embed_dim = embed_dim
        
        # Modify ddconfig for latent processing
        latent_ddconfig = ddconfig.copy()
        
        self.encoder = Encoder(**latent_ddconfig)
        self.decoder = Decoder(**latent_ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.beta = self.loss.beta
        
        assert latent_ddconfig["double_z"]

        # Calculate bottleneck spatial dimensions
        bottleneck_resolution = latent_ddconfig["resolution"] // (2 ** (len(latent_ddconfig["ch_mult"])-1))    
        spatial_size = bottleneck_resolution * bottleneck_resolution

        # Create custom layers for flattening to 1D latent space
        self.quant_conv = torch.nn.Sequential(
            torch.nn.Flatten(),  # Flatten spatial dimensions to 1D
            torch.nn.Linear(2*latent_ddconfig["z_channels"] * spatial_size, 2*embed_dim)  # Project to flat latent dim (2* for mean/logvar)
        )
        
        # Create custom layers for reshaping from 1D latent back to spatial
        self.post_quant_conv = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, latent_ddconfig["z_channels"] * spatial_size),  # Project from flat latent to spatial
            torch.nn.Unflatten(1, (latent_ddconfig["z_channels"], bottleneck_resolution, bottleneck_resolution))  # Reshape to spatial
        )
        
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.save_hyperparameters()
        self.learning_rate = 1e-4

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def save_checkpoint(self, path, optimizer=None):
        """
        Save the current model state to a checkpoint file.
        
        Args:
            path (str): Path where to save the checkpoint
            optimizer (torch.optim.Optimizer, optional): Optimizer state to save
        """
        if optimizer is not None:
            checkpoint = {
                'state_dict': self.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': self.hparams
            }
        else:
            checkpoint = {
                'state_dict': self.state_dict(),
                'config': self.hparams
            }
        
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior
    
    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec
    
    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior
    
    def get_input(self, batch):
        x = batch
        # Latents are already in the right format (B, C, H, W), so we just ensure correct dtype
        return x.to(memory_format=torch.contiguous_format).float()

    def training_step(self, batch, batch_idx):
        inputs = self.get_input(batch)
        reconstructions, posterior = self(inputs)
        
        # Simple combined loss (reconstruction + KL divergence)
        rec_loss = F.mse_loss(reconstructions, inputs)
        kl_loss = posterior.kl().mean()
        
        # Total loss with weighting
        total_loss = rec_loss + self.beta * kl_loss

        # Log losses
        self.log("rec_loss", rec_loss, prog_bar=True)
        self.log("kl_loss", kl_loss, prog_bar=True)
        self.log("total_loss", total_loss, prog_bar=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch)
        reconstructions, posterior = self(inputs)
        
        rec_loss = F.mse_loss(reconstructions, inputs)
        kl_loss = posterior.kl().mean()
        
        total_loss = rec_loss + self.beta * kl_loss
        
        # Log losses
        self.log("val_rec_loss", rec_loss, prog_bar=True)
        self.log("val_kl_loss", kl_loss, prog_bar=True)
        self.log("val_total_loss", total_loss, prog_bar=True)
        
        return total_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def get_last_layer(self):
        return self.decoder.conv_out if hasattr(self.decoder, 'conv_out') else None
    
    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        latents = self.get_input(batch)
        latents = latents.to(self.device)
        
        if not only_inputs:
            # Get reconstructions
            latents_rec, posterior = self(latents)
        
            # Log encoded features (before quantization)
            h = self.encoder(latents)  # Get intermediate encoding before distribution
            log["encoder_features"] = h  # These are feature maps
            
            # Generate random samples from latent space
            z = posterior.sample()
            log["latent_sample"] = z  # The actual latent code
            random_samples = self.decode(torch.randn_like(z))
            
            # Add to log dictionary
            log["reconstructions"] = latents_rec
            log["samples"] = random_samples
            log["rec_error"] = torch.abs(latents - latents_rec)
            
        log["inputs"] = latents
        return log