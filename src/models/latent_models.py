import torch
import os
from torch import nn
import torch.distributed as dist
import pytorch_lightning as pl
from torchvision.utils import make_grid

# Import Stable Diffusion VAE decoder for pixel-space losses
from diffusers import AutoencoderKL

from src.models.modules.autoencoder import Encoder, Decoder
from src.models.modules.distributions import DiagonalGaussianDistribution
from src.models.modules.quantize import VectorQuantizer2 as VectorQuantizer
from src.models.modules.utils import instantiate_from_config
from src.models.modules.losses import *
from abc import ABC, abstractmethod


class LatentModel(pl.LightningModule, ABC):
    """
    Abstract base class for latent models.
    Provides a common interface for encoding, decoding, and training latent models.
    """
    def __init__(
            self,
            ddconfig,
            lossconfig,
            learning_rate=1e-4, 
            sd_vae_path=None,
            fid_instance=None,
            spectral_instance=None
        ):
        """
        Initialize the LatentModel.
        Args:
            ddconfig: Configuration for the encoder and decoder.
            lossconfig: Configuration for the loss function.
            learning_rate: Learning rate for the optimizer.
            sd_vae_path: Path to the Stable Diffusion VAE model for perceptual loss.
            fid_instance: Instance of the FID metric for evaluation.
            spectral_instance: Instance of the Spectral metric for evaluation.
        """
        super().__init__()

        # Store learning rate
        self.learning_rate = learning_rate

        # Loss setup
        self.loss = instantiate_from_config(lossconfig)

        # Stable diffusion VAE for perceptual loss
        if sd_vae_path is not None:
            self.sd_vae_path = sd_vae_path
            self.sd_vae = AutoencoderKL.from_pretrained(sd_vae_path, subfolder="vae")
            self.sd_vae.eval()
            self.sd_vae.requires_grad_(False)
        else:
            self.sd_vae = None

        # Manual optimization if Discriminator is used in loss function
        if isinstance(self.loss, (LPIPSWithDiscriminator, VAEWithDiscriminator, 
                                  VQLPIPSWithDiscriminator, AutoencoderLPIPSWithDiscriminator)):
            self.automatic_optimization = False
        else:
            self.automatic_optimization = True

        # FID metric
        self.fid_instance = fid_instance
        self.track_fid = bool(self.fid_instance is not None)

        # Spectral metric
        self.spectral_instance = spectral_instance
        self.track_spectral = bool(self.spectral_instance is not None)

        # Save hyperparameters
        self.save_hyperparameters(
            "ddconfig",
            "lossconfig",
            "sd_vae_path",
            "ckpt_path",
            "learning_rate",
        )

    def init_from_ckpt(self, path, ignore_keys=list()):
        """
        Initialize model weights from a checkpoint.
        Args:
            path (str): Path to the checkpoint file.
            ignore_keys (list): List of keys to ignore when loading weights.
        """
        # Load state dict from checkpoint
        sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        keys = list(sd.keys())
        # Remove keys that start with any of the ignore keys
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored model state from {path}")

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

    @abstractmethod
    def encode(self, x):
        """Encodes a latent tensor to a lower dimension."""
        pass

    @abstractmethod
    def decode(self, z):
        """Decodes a latent tensor to a higher dimension."""
        pass

    @abstractmethod
    def forward(self, x, return_only_recon=False):
        """Encodes and decodes a latent tensor."""
        pass

    def get_input(self, batch):
        """Get the input tensor from the batch and ensure its correct dtype."""
        return batch.to(memory_format=torch.contiguous_format).float()

    @torch.no_grad()
    def latents_to_images(self, latents):
        """
        Decode Stable-Diffusion latents to pixel space.
        Args:
            latents (torch.Tensor): Latent tensor of shape (B, C, H, W).
        Returns:
            torch.Tensor: Reconstructed image tensor of shape (B, C, H, W).
        Raises:
            RuntimeError: If sd_vae is not initialized.
        """
        if self.sd_vae is None:
            raise RuntimeError("latents_to_images() called but sd_vae is not initialized.")
        # Move decoder to correct device if necessary
        self.sd_vae.to(latents.device)
        images = self.sd_vae.decode(latents).sample

        # Clamp the images to the range [-1, 1]
        images = torch.clamp(images, -1, 1)

        return images

    def configure_optimizers(self):
        """Configures the optimizer for the model."""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    @abstractmethod
    def training_step(self, batch, batch_idx):
        """Training step for the model."""
        pass

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        """Validation step for the model."""
        pass

    def on_train_start(self):
        """Grab a fixed mini-batch of latents for logging."""
        if self.global_rank == 0:
            # Grab the first validation batch and keep up to 16 examples
            val_batch = next(iter(self.trainer.datamodule.val_dataloader()))
            latents = val_batch[0] if isinstance(val_batch, (list, tuple)) else val_batch
            # Preserve batch dimension (B,C,H,W) and move to the right device
            self.fixed_latents = latents[:16].to(self.device)

    @torch.no_grad()
    def _log_reconstructions(self):
        """Log reconstructed images for visualization."""
        # Ensure tensor is on the correct device and has a batch dimension
        inputs = self.fixed_latents.to(self.device)
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(0)
        recons = self(inputs, return_only_recon=True)

        # Decode latents to images
        inputs_img = self.latents_to_images(inputs)
        recons_img = self.latents_to_images(recons)

        # Create grid of images
        grid = make_grid(torch.cat([inputs_img, recons_img], dim=0), nrow=8, normalize=True)

        # Log the grid of images
        self.logger.experiment.add_image("reconstructions", grid, self.global_step)

    def on_validation_epoch_start(self):
        """Initialize list to store reconstructed images for metric calculation."""
        if self.track_fid or self.track_spectral:
            # Initialize list to store reconstructed images for metric calculation
            self._val_recons_img = []

        return super().on_validation_epoch_start()
    
    def on_validation_epoch_end(self):
        """Log validation metrics and calculate FID/Spectral scores if applicable."""
        # Skip logging if sanity checking
        if self.trainer.sanity_checking:
            return super().on_validation_epoch_end()
        
        # -- Validation loss -------------------------------------- #
        if self.global_rank == 0:
            print(f"Epoch {self.current_epoch}:")
            print(f"~ Validation loss: {self.trainer.callback_metrics['val/total_loss']}")

        # -- Get reconstructed images ----------------------------- #
        if self.track_fid or self.track_spectral:
            # Concatenate all batches of reconstructed images (ensure CPU)
            recon_img = torch.cat(self._val_recons_img, dim=0).cpu()
            torch.cuda.empty_cache()

            # Gather from all ranks, but keep on CPU
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, recon_img)
            recon_img = torch.cat(gathered, dim=0)

        # -- FID & Spectral score --------------------------------- #
        if self.global_rank == 0:
            
            if self.track_fid:
                # Compute FID score on the reconstructed images
                fid_score = self.fid_instance.compute_score_from_data(recon_img, eps=1E-6)

                # Log FID score
                print(f"~ FID score: {fid_score}")
                self.log("val/fid_score", fid_score, prog_bar=False, on_step=False, on_epoch=True, sync_dist=False)

            if self.track_spectral:
                # Compute Spectral score on the reconstructed images
                spectral_score = self.spectral_instance.compute_score_from_data(recon_img, eps=1E-6)

                # Log Spectral score
                print(f"~ Spectral score: {spectral_score}")
                self.log("val/spectral_score", spectral_score, prog_bar=False, on_step=False, on_epoch=True, sync_dist=False)

        # -- Log reconstructions ---------------------------------- #
        if self.global_rank == 0:
            # Log reconstructed images
            self._log_reconstructions()

        # -- Clean up --------------------------------------------- #
        if self.track_fid or self.track_spectral:
            # Clear the list of reconstructed images
            self._val_recons_img = []

        super().on_validation_epoch_end()


class LatentVAE(LatentModel):
    def __init__(
            self,
            ddconfig,
            lossconfig,
            learning_rate=1e-4,
            sd_vae_path=None,
            ckpt_path=None,
            ignore_keys=[],
            fid_instance=None,
            spectral_instance=None,
        ):
        """
        LatentVAE is a modified version of the AutoencoderKL that processes latents instead of images.
        It uses the same encoder and decoder architecture but is designed to work in the latent space
        of a diffusion model.
        Args:
            ddconfig: Configuration for the encoder and decoder.
            lossconfig: Configuration for the loss function.
            learning_rate: Learning rate for the optimizer.
            sd_vae_path: Path to the Stable Diffusion VAE model for perceptual loss.
            ckpt_path: Path to a checkpoint to load weights from.
            ignore_keys: Keys to ignore when loading weights from the checkpoint.
            fid_instance: Instance of the FID metric for evaluation.
            spectral_instance: Instance of the Spectral metric for evaluation.
        """
        # Setup general stuff for latent models
        super().__init__(
            ddconfig=ddconfig,
            lossconfig=lossconfig,
            learning_rate=learning_rate,
            sd_vae_path=sd_vae_path,
            fid_instance=fid_instance,
            spectral_instance=spectral_instance,
        )
        
        # Ensure it is a VAE
        assert ddconfig["double_z"]

        # Create encoder/decoder
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)

        # Get parameters from the config
        embed_dim = ddconfig['embed_dim']
        z_channels = ddconfig["z_channels"]
        ch_mult = ddconfig["ch_mult"]
        resolution = ddconfig["resolution"]

        # Calculate bottleneck spatial dimensions
        bottleneck_resolution = resolution // (2 ** (len(ch_mult)-1))
        spatial_size = bottleneck_resolution * bottleneck_resolution

        # Create custom layers for flattening to 1D latent space
        self.quant_conv = torch.nn.Sequential(
            torch.nn.Flatten(),  # Flatten spatial dimensions to 1D
            torch.nn.Linear(2*z_channels * spatial_size, 2*embed_dim)  # Project to flat latent dim (2* for mean/logvar)
        )
        
        # Create custom layers for reshaping from 1D latent back to spatial
        self.post_quant_conv = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, z_channels * spatial_size),  # Project from flat latent to spatial
            torch.nn.Unflatten(1, (z_channels, bottleneck_resolution, bottleneck_resolution))  # Reshape to spatial
        )

        # Initialize from checkpoint if provided
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def encode(self, x):
        """
        Encode a latent tensor to a lower dimension.
        Args:
            x (torch.Tensor): Input latent tensor of shape (B, C, H, W).
        Returns:
            DiagonalGaussianDistribution: Posterior distribution of the latent space.
        """
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior
    
    def decode(self, z):
        """
        Decode a latent tensor to a higher dimension.
        Args:
            z (torch.Tensor): Input latent tensor of shape (B, C, H, W).
        Returns:
            torch.Tensor: Reconstructed image tensor of shape (B, C, H, W).
        """
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec
    
    def forward(self, input, sample_posterior=True, return_only_recon=False):
        """
        Encode and decode a latent tensor.
        Args:
            input (torch.Tensor): Input latent tensor of shape (B, C, H, W).
            sample_posterior (bool): Whether to sample from the posterior distribution.
            return_only_recon (bool): Whether to return only the reconstructed image.
        Returns:
            torch.Tensor: Reconstructed image tensor of shape (B, C, H, W).
            DiagonalGaussianDistribution: Posterior distribution of the latent space.
        """
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        if return_only_recon:
            return dec
        else:
            return dec, posterior
    
    def training_step(self, batch, batch_idx):
        """
        Training step for the latent VAE.
        Args:
            batch (torch.Tensor): Input batch tensor.
            batch_idx (int): Index of the current batch.
        Returns:
            torch.Tensor: Total loss for the current batch.
        """
        # -- 1. Get data ------------------------------------------ #
        # Get inputs & reconstructions
        inputs = self.get_input(batch)
        recons, posterior = self(inputs)

        # Decode latents to images for perceptual loss
        if isinstance(self.loss, (LPIPSWithDiscriminator, VAEWithDiscriminator)):
            with torch.no_grad():
                inputs_img = self.latents_to_images(inputs)
                recons_img = self.latents_to_images(recons)

        # -- 2. Run training step based on the loss function ------ #

        if isinstance(self.loss, LPIPSWithDiscriminator):
            # -- 2A. LPIPS with Discriminator loss ---------------- #

            # Grab the two optimizers
            opt_g, opt_d = self.optimizers()

            # Generator / VAE update (optimizer_idx = 0)
            loss_g, log_g = self.loss(
                inputs, recons,
                inputs_img, recons_img,
                posterior,
                optimizer_idx=0,
                global_step=self.global_step,
                split="train"
            )

            opt_g.zero_grad()
            self.manual_backward(loss_g)
            opt_g.step()

            # Discriminator update (optimizer_idx = 1)
            # detach everything that should not receive grads
            loss_d, log_d = self.loss(
                inputs.detach(), recons.detach(),
                inputs_img.detach(), recons_img.detach(),
                posterior.detach(),
                optimizer_idx=1,
                global_step=self.global_step,
                split="train"
            )

            opt_d.zero_grad()
            self.manual_backward(loss_d)
            opt_d.step()

            # Summarize losses
            log_dict = {**log_g, **log_d}
            total_loss = loss_g + loss_d

        elif isinstance(self.loss, VAEWithDiscriminator):
            # -- 2B. VAE with Discriminator loss ------------------ #

            # Grab the two optimizers
            opt_g, opt_d = self.optimizers()

            # Generator / VAE update (optimizer_idx = 0)
            loss_g, log_g = self.loss(
                inputs=inputs,
                recons=recons,
                posterior=posterior,
                optimizer_idx=0,
                global_step=self.global_step,
                split="train"
            )

            opt_g.zero_grad()
            self.manual_backward(loss_g)  
            opt_g.step()

            # Discriminator update (optimizer_idx = 1)
            loss_d, log_d = self.loss(
                inputs=inputs.detach(),
                recons=recons.detach(),
                posterior=posterior.detach(),
                optimizer_idx=1,
                global_step=self.global_step,
                split="train"
            )
            opt_d.zero_grad()
            self.manual_backward(loss_d)
            opt_d.step()

            # Summarize losses
            log_dict = {**log_g, **log_d}
            total_loss = loss_g + loss_d
            
        elif isinstance(self.loss, SimpleVAELoss):
            # -- 2C. Simple VAE loss ------------------------------ #

            # Compute loss
            total_loss, log_dict = self.loss(
                inputs=inputs,
                recons=recons,
                posterior=posterior,
                split="train"
            )
        
        # -- 3. Logging ------------------------------------------- #
        self.log_dict(log_dict, prog_bar=False, on_step=True, on_epoch=False, sync_dist=True)
        self.log("train/total_loss", total_loss, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)

        return total_loss    
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step for the latent VAE.
        Args:
            batch (torch.Tensor): Input batch tensor.
            batch_idx (int): Index of the current batch.
        Returns:
            torch.Tensor: Total loss for the current batch.
        """
        # -- 1. Get data ------------------------------------------ #
        # Get inputs & reconstructions
        inputs = self.get_input(batch)
        recons, posterior = self(inputs)

        # Decode latents to images for perceptual loss or FID/Spectral score
        if isinstance(self.loss, (LPIPSWithDiscriminator, VAEWithDiscriminator)) or self.track_fid or self.track_spectral:
            with torch.no_grad():
                inputs_img = self.latents_to_images(inputs)
                recons_img = self.latents_to_images(recons)
            # Store reconstructed images for FID or Spectral calculation
            if self.track_fid or self.track_spectral:
                self._val_recons_img.append(recons_img.cpu())

        # -- 2. Run training step based on the loss function ------ #

        if isinstance(self.loss, LPIPSWithDiscriminator):
            # -- 2A. LPIPS with Discriminator loss ---------------- #
            
            # Generator / VAE loss (optimizer_idx = 0)
            loss_g, log_g = self.loss(
                inputs, recons,
                inputs_img, recons_img,
                posterior,
                optimizer_idx=0,
                global_step=self.global_step,
                split="val"
            )

            # Discriminator loss (optimizer_idx = 1)
            loss_d, log_d = self.loss(
                inputs.detach(), recons.detach(),
                inputs_img.detach(), recons_img.detach(),
                posterior.detach(),
                optimizer_idx=1,
                global_step=self.global_step,
                split="val"
            )

            # Summarize losses
            log_dict = {**log_g, **log_d}
            total_loss = loss_g + loss_d

        elif isinstance(self.loss, VAEWithDiscriminator):
            # -- 2B. VAE with Discriminator loss ------------------ #

            # Generator / VAE update (optimizer_idx = 0)
            loss_g, log_g = self.loss(
                inputs=inputs,
                recons=recons,
                posterior=posterior,
                optimizer_idx=0,
                global_step=self.global_step,
                split="val"
            )

            # Discriminator update (optimizer_idx = 1)
            loss_d, log_d = self.loss(
                inputs=inputs.detach(),
                recons=recons.detach(),
                posterior=posterior.detach(),
                optimizer_idx=1,
                global_step=self.global_step,
                split="val"
            )

            # Summarize losses
            log_dict = {**log_g, **log_d}
            total_loss = loss_g + loss_d
            
        elif isinstance(self.loss, SimpleVAELoss):
            # -- 2C. Simple VAE loss ------------------------------ #

            # Compute loss
            total_loss, log_dict = self.loss(
                inputs=inputs,
                recons=recons,
                posterior=posterior,
                split="val"
            )

        # -- 3. Logging ------------------------------------------- #
        self.log_dict(log_dict, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/total_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        return total_loss

    def configure_optimizers(self):
        """
        Configure the optimizers for the model.
        Returns:
            list: List of optimizers.
        """
        if isinstance(self.loss, (VAEWithDiscriminator, LPIPSWithDiscriminator)):
            # Get relevant parameters for VAE and discriminator
            d_params = list(self.loss.discriminator.parameters())
            d_ids = set(id(p) for p in d_params)
            g_params = [p for p in self.parameters() if id(p) not in d_ids]

            # Learning rates
            lr_g = self.learning_rate
            lr_d = 4 * lr_g

            # Use two optimizers for VAE and discriminator
            # First optimizer (generator / VAE)
            opt_g= torch.optim.Adam(
                g_params,
                lr=lr_g, 
                betas=(0.5, 0.999)
            )

            # Second optimizer (discriminator)
            opt_d = torch.optim.Adam(
                d_params,
                lr=lr_d, 
                betas=(0.0, 0.999)
            )

            # Return both optimizers
            return [opt_g, opt_d]

        elif isinstance(self.loss, SimpleVAELoss):
            # Use a single optimizer for VAE
            return torch.optim.Adam(
                self.parameters(), 
                lr=self.learning_rate
            )


class LatentVQVAE(LatentModel):
    def __init__(
            self,
            ddconfig,
            lossconfig,
            learning_rate=1e-4,
            sd_vae_path=None,
            ckpt_path=None,
            ignore_keys=[],
            fid_instance=None,
            spectral_instance=None,
        ):
        """
        LatentVQVAE to process latents instead of images.
        Args:
            ddconfig: Configuration for the encoder and decoder.
            lossconfig: Configuration for the loss function.
            learning_rate: Learning rate for the optimizer.
            sd_vae_path: Path to the Stable Diffusion VAE model for perceptual loss.
            ckpt_path: Path to a checkpoint to load weights from.
            ignore_keys: Keys to ignore when loading weights from the checkpoint.
            fid_instance: Instance of the FID metric for evaluation.
            spectral_instance: Instance of the Spectral metric for evaluation.
        """
        # Setup general stuff for latent models
        super().__init__(
            ddconfig=ddconfig,
            lossconfig=lossconfig,
            learning_rate=learning_rate,
            sd_vae_path=sd_vae_path,
            fid_instance=fid_instance,
            spectral_instance=spectral_instance,
        )

        # Create encoder/decoder
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)

        # Get parameters from the config
        n_embed = ddconfig["n_embed"]
        embed_dim = ddconfig["embed_dim"]
        z_channels = ddconfig["z_channels"]
        remap = ddconfig["remap"]
        sane_index_shape = ddconfig["sane_index_shape"]
        ch_mult = ddconfig["ch_mult"]
        resolution = ddconfig["resolution"]

        # Calculate bottleneck spatial dimensions
        bottleneck_resolution = resolution // (2 ** (len(ch_mult)-1))
        assert bottleneck_resolution * bottleneck_resolution == embed_dim, \
            f"`embed_dim` ({embed_dim}) must be equal to bottleneck_resolution^2"
        self.latent_dim = (bottleneck_resolution, bottleneck_resolution, z_channels)

        # Setup quantizer (B, z_channels, H, W)
        self.quant_conv = torch.nn.Conv2d(z_channels, z_channels, 1)
        self.quantize = VectorQuantizer(n_embed, z_channels, beta=0.05,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.post_quant_conv = torch.nn.Conv2d(z_channels, z_channels, 1)

        # Initialize from checkpoint if provided
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def encode(self, x):
        """
        Encode a latent tensor to a lower dimension.
        Args:
            x (torch.Tensor): Input latent tensor of shape (B, C, H, W).
        Returns:
            tuple: Quantized tensor, embedding loss, and indices.
        """
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, (_,_,ind) = self.quantize(h)
        ind = ind.reshape(ind.shape[0], -1) # Flatten indices to (B, embed_dim)
        return quant, emb_loss, ind

    def encode_to_prequant(self, x):
        """
        Encode a latent tensor to a lower dimension without quantization.
        Args:
            x (torch.Tensor): Input latent tensor of shape (B, C, H, W).
        Returns:
            torch.Tensor: Encoded tensor before quantization.
        """
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, quant):
        """
        Decode a quantized tensor to a higher dimension.
        Args:
            quant (torch.Tensor): Input quantized tensor of shape (B, z_channels, H, W).
        Returns:
            torch.Tensor: Reconstructed tensor of shape (B, C, H, W).
        """
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        """
        Decode a tensor of indices to a higher dimensional representation.
        Args:
            code_b (torch.Tensor): Tensor containing indices of shape (B, embed_dim) or (embed_dim).
        Returns:
            torch.Tensor: Reconstructed tensor of shape (B, C, H, W).
        """
        if code_b.dim() == 1:
            # If code_b is a single vector, reshape it to (1, embed_dim)
            code_b = code_b.unsqueeze(0)
        quant_b = self.quantize.get_codebook_entry(code_b, shape=(code_b.shape[0], *self.latent_dim))
        dec = self.decode(quant_b)
        return dec
    
    def forward(self, input, return_pred_indices=False, return_only_recon=False):
        """
        Encode and decode a latent tensor.
        Args:
            input (torch.Tensor): Input latent tensor of shape (B, C, H, W).
            return_pred_indices (bool): Whether to return the predicted indices.
            return_only_recon (bool): Whether to return only the reconstructed image.
        Returns:
            tuple: Reconstructed tensor, embedding loss, and predicted indices (if requested).
        """
        quant, diff, ind = self.encode(input)
        dec = self.decode(quant)
        if return_only_recon:
            return dec
        elif return_pred_indices:
            return dec, diff, ind
        else:
            return dec, diff

    def training_step(self, batch, batch_idx):
        """
        Training step for the latent VQVAE.
        Args:
            batch (torch.Tensor): Input batch tensor.
            batch_idx (int): Index of the current batch.
        Returns:
            torch.Tensor: Total loss for the current batch.
        """
        # -- 1. Get data ------------------------------------------ #
        # Get inputs & reconstructions
        inputs = self.get_input(batch)
        recons, qloss, ind = self(inputs, return_pred_indices=True)

        # Decode latents to images for perceptual loss
        if isinstance(self.loss, VQLPIPSWithDiscriminator):
            with torch.no_grad():
                inputs_img = self.latents_to_images(inputs)
                recons_img = self.latents_to_images(recons)

        # -- 2. Run training step based on the loss function ------ #

        if isinstance(self.loss, VQLPIPSWithDiscriminator):
            # -- 2A. LPIPS with Discriminator loss ---------------- #

            # Grab the two optimizers
            opt_g, opt_d = self.optimizers()

            # Generator / VAE update (optimizer_idx = 0)
            loss_g, log_g = self.loss(
                inputs, recons,
                inputs_img, recons_img,
                qloss,
                optimizer_idx=0,
                global_step=self.global_step,
                predicted_indices=ind,
                split="train"
            )

            opt_g.zero_grad()
            self.manual_backward(loss_g)
            opt_g.step()

            # Discriminator update (optimizer_idx = 1)
            # detach everything that should not receive grads
            loss_d, log_d = self.loss(
                inputs.detach(), recons.detach(),
                inputs_img.detach(), recons_img.detach(),
                qloss.detach(),
                optimizer_idx=1,
                global_step=self.global_step,
                split="train"
            )

            opt_d.zero_grad()
            self.manual_backward(loss_d)
            opt_d.step()

            # Summarize losses
            log_dict = {**log_g, **log_d}
            total_loss = loss_g + loss_d

        elif isinstance(self.loss, SimpleVQVAELoss):
            # -- 2B. Simple VQ-VAE loss --------------------------- #

            # Compute loss
            total_loss, log_dict = self.loss(
                inputs=inputs,
                recons=recons,
                qloss=qloss,
                split="train"
            )

        # -- 3. Logging ------------------------------------------- #
        self.log_dict(log_dict, prog_bar=False, on_step=True, on_epoch=False, sync_dist=True)
        self.log("train/total_loss", total_loss, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the latent VQVAE.
        Args:
            batch (torch.Tensor): Input batch tensor.
            batch_idx (int): Index of the current batch.
        Returns:
            torch.Tensor: Total loss for the current batch.
        """
        # -- 1. Get data ------------------------------------------ #
        # Get inputs & reconstructions
        inputs = self.get_input(batch)
        recons, qloss, ind = self(inputs, return_pred_indices=True)

        # Decode latents to images for perceptual loss or FID/Spectral score
        if isinstance(self.loss, LPIPSWithDiscriminator) or self.track_fid or self.track_spectral:
            with torch.no_grad():
                inputs_img = self.latents_to_images(inputs)
                recons_img = self.latents_to_images(recons)
            # Store reconstructed images for FID or Spectral calculation
            if self.track_fid or self.track_spectral:
                self._val_recons_img.append(recons_img.cpu())

        # -- 2. Run training step based on the loss function ------ #

        if isinstance(self.loss, VQLPIPSWithDiscriminator):
            # -- 2A. LPIPS with Discriminator loss ---------------- #
            
            # Generator / VAE loss (optimizer_idx = 0)
            loss_g, log_g = self.loss(
                inputs, recons,
                inputs_img, recons_img,
                qloss,
                optimizer_idx=0,
                global_step=self.global_step,
                predicted_indices=ind,
                split="val"
            )

            # Discriminator loss (optimizer_idx = 1)
            loss_d, log_d = self.loss(
                inputs.detach(), recons.detach(),
                inputs_img.detach(), recons_img.detach(),
                qloss.detach(),
                optimizer_idx=1,
                global_step=self.global_step,
                split="val"
            )

            # Summarize losses
            log_dict = {**log_g, **log_d}
            total_loss = loss_g + loss_d

        elif isinstance(self.loss, SimpleVQVAELoss):
            # -- 2B. Simple VQ-VAE loss --------------------------- #

            # Compute loss
            total_loss, log_dict = self.loss(
                inputs=inputs,
                recons=recons,
                qloss=qloss,
                split="val"
            )

        # -- 3. Logging ------------------------------------------- #
        self.log_dict(log_dict, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/total_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        return total_loss

    def configure_optimizers(self):
        """
        Configure the optimizers for the model.
        Returns:
            list: List of optimizers.
        """
        if isinstance(self.loss, (VQLPIPSWithDiscriminator)):
            # Get relevant parameters for VAE and discriminator
            d_params = list(self.loss.discriminator.parameters())
            d_ids = set(id(p) for p in d_params)
            g_params = [p for p in self.parameters() if id(p) not in d_ids]

            # Learning rates
            lr_g = self.learning_rate
            lr_d = 4 * lr_g

            # Use two optimizers for VAE and discriminator
            # First optimizer (generator / VAE)
            opt_g = torch.optim.Adam(
                g_params,
                lr=lr_g, 
                betas=(0.5, 0.999)
            )

            # Second optimizer (discriminator)
            opt_d = torch.optim.Adam(
                d_params,
                lr=lr_d, 
                betas=(0.0, 0.999)
            )

            # Return both optimizers
            return [opt_g, opt_d]
        
        elif isinstance(self.loss, SimpleVQVAELoss):
            # Use a single optimizer for VQ-VAE
            return torch.optim.Adam(
                self.parameters(), 
                lr=self.learning_rate
            )


class LatentVQVAE2(LatentModel):
    """
    Hierarchical two-level VQ-VAE-2 that operates directly in the Stable-Diffusion
    latent space.  A top-level code captures global structure, while a bottom-level
    code refines fine detail, mirroring the original VQ-VAE-2 design.
    Expect `ddconfig` to contain two nested dicts: `bottom` and `top`.
    """
    def __init__(
            self,
            ddconfig,
            lossconfig,
            learning_rate: float = 1e-4,
            sd_vae_path: str | None = None,
            ckpt_path: str | None = None,
            ignore_keys: list = [],
            fid_instance=None,
            spectral_instance=None,
        ):
        """
        LatentVQVAE2 to process latents instead of images.
        Args:
            ddconfig: Configuration for the encoder and decoder.
            lossconfig: Configuration for the loss function.
            learning_rate: Learning rate for the optimizer.
            sd_vae_path: Path to the Stable Diffusion VAE model for perceptual loss.
            ckpt_path: Path to a checkpoint to load weights from.
            ignore_keys: Keys to ignore when loading weights from the checkpoint.
            fid_instance: Instance of the FID metric for evaluation.
            spectral_instance: Instance of the Spectral metric for evaluation.
        """
        super().__init__(
            ddconfig=ddconfig,
            lossconfig=lossconfig,
            learning_rate=learning_rate,
            sd_vae_path=sd_vae_path,
            fid_instance=fid_instance,
            spectral_instance=spectral_instance,
        )
        assert "bottom" in ddconfig and "top" in ddconfig, (
            "For VQ-VAE-2, `ddconfig` must have `bottom` and `top` keys."
        )
        bottom_cfg = ddconfig["bottom"]
        top_cfg    = ddconfig["top"]

        # -- Encoders & Decoders ---------------------------------- #
        self.encoder_bottom = Encoder(**bottom_cfg)
        self.encoder_top    = Encoder(**top_cfg)

        self.decoder_top    = Decoder(**top_cfg)     # Upsamples top code to bottom res.
        self.decoder_bottom = Decoder(**bottom_cfg)  # Final reconstruction

        # -- Latent Dimensions ------------------------------------ #
        # Get parameters from the config
        embed_dim_bottom = bottom_cfg["embed_dim"]
        embed_dim_top    = top_cfg["embed_dim"]
        z_channels_bottom = bottom_cfg["z_channels"]
        z_channels_top    = top_cfg["z_channels"]
        # Calculate bottleneck spatial dimensions
        bottleneck_resolution_bottom = bottom_cfg["resolution"] // (2 ** (len(bottom_cfg["ch_mult"]) - 1))
        bottleneck_resolution_top    = top_cfg["resolution"] // (2 ** (len(top_cfg["ch_mult"]) - 1))
        assert bottleneck_resolution_bottom * bottleneck_resolution_bottom == embed_dim_bottom, \
            f"`embed_dim_bottom` ({embed_dim_bottom}) must be equal to bottleneck_resolution_bottom^2"
        assert bottleneck_resolution_top * bottleneck_resolution_top == embed_dim_top, \
            f"`embed_dim_top` ({embed_dim_top}) must be equal to bottleneck_resolution_top^2"
        self.latent_dim_bottom = (bottleneck_resolution_bottom, bottleneck_resolution_bottom, z_channels_bottom)
        self.latent_dim_top    = (bottleneck_resolution_top, bottleneck_resolution_top, z_channels_top)

        # -- Vector‑Quantizers ------------------------------------ #
        # Top level
        self.quant_conv_top = torch.nn.Conv2d(top_cfg["z_channels"], top_cfg["z_channels"], 1)
        self.quantize_top   = VectorQuantizer(top_cfg["n_embed"], top_cfg["z_channels"], beta=0.05,
                                                remap=top_cfg["remap"], sane_index_shape=top_cfg["sane_index_shape"])
        self.post_quant_conv_top = torch.nn.Conv2d(top_cfg["z_channels"], top_cfg["z_channels"], 1)

        # Bottom level
        self.quant_conv_bottom = torch.nn.Conv2d(bottom_cfg["z_channels"], bottom_cfg["z_channels"], 1)
        self.quantize_bottom   = VectorQuantizer(bottom_cfg["n_embed"], bottom_cfg["z_channels"], beta=0.05,
                                                  remap=bottom_cfg["remap"], sane_index_shape=bottom_cfg["sane_index_shape"])
        self.post_quant_conv_bottom = torch.nn.Conv2d(bottom_cfg["z_channels"], bottom_cfg["z_channels"], 1)

        # Optionally restore weights
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def encode(self, x):
        """
        Encode an input tensor to produce top and bottom codes.
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
        Returns:
            tuple: Quantized bottom code, quantized top code, total embedding loss,
                   and concatenated indices for both codes.
        """
        # Bottom-level features
        h_b = self.encoder_bottom(x)                # (B, C_b, H_b, W_b)

        # Top-level features
        h_t = self.encoder_top(h_b)                 # (B, C_t, H_t, W_t)
        h_t = self.quant_conv_top(h_t)
        q_t, diff_t, (_, _, ind_t) = self.quantize_top(h_t)  # (B, C_t, H_t, W_t)

        # Decode top code back to bottom resolution (conditioning)
        d_t = self.decoder_top(self.post_quant_conv_top(q_t))

        # Residual connection as in original paper
        h_b = self.quant_conv_bottom(h_b + d_t)
        q_b, diff_b, (_, _, ind_b) = self.quantize_bottom(h_b)  # (B, C_b, H_b, W_b)

        diff_total = diff_t + diff_b
        # Flatten & concatenate indices for logging/usage
        ind = torch.cat(
            [ind_t.view(ind_t.size(0), -1),
             ind_b.view(ind_b.size(0), -1)],
            dim=1
        )
        return q_b, q_t, diff_total, ind

    def decode(self, q_b, q_t):
        """
        Decode the quantized bottom and top codes to reconstruct the original input.
        Args:
            q_b (torch.Tensor): Quantized bottom code of shape (B, C_b, H_b, W_b).
            q_t (torch.Tensor): Quantized top code of shape (B, C_t, H_t, W_t).
        Returns:
            torch.Tensor: Reconstructed tensor of shape (B, C, H, W).
        """
        d_t = self.decoder_top(self.post_quant_conv_top(q_t))
        x   = self.decoder_bottom(self.post_quant_conv_bottom(q_b) + d_t)
        return x
    
    def decode_code(self, code_b, code_t):
        """
        Decode a tensor of indices to a higher dimensional representation.
        Args:
            code_b (torch.Tensor): Tensor containing bottom indices of shape (B, embed_dim).
            code_t (torch.Tensor): Tensor containing top indices of shape (B, embed_dim).
        Returns:
            torch.Tensor: Reconstructed tensor of shape (B, C, H, W).
        """
        if code_b.dim() == 1:
            # If code_b is a single vector, reshape it to (1, embed_dim)
            code_b = code_b.unsqueeze(0)
        if code_t.dim() == 1:
            # If code_t is a single vector, reshape it to (1, embed_dim)
            code_t = code_t.unsqueeze(0)
        
        quant_b = self.quantize_bottom.get_codebook_entry(code_b, shape=(code_b.shape[0], *self.latent_dim_bottom))
        quant_t = self.quantize_top.get_codebook_entry(code_t, shape=(code_t.shape[0], *self.latent_dim_top))
        dec = self.decode(quant_b, quant_t)
        return dec

    def forward(self, input, return_pred_indices=False, return_only_recon=False):
        """
        Encode and decode an input tensor.
        Args:
            input (torch.Tensor): Input tensor of shape (B, C, H, W).
            return_pred_indices (bool): Whether to return the predicted indices.
            return_only_recon (bool): Whether to return only the reconstructed image.
        Returns:
            tuple: Reconstructed tensor, total embedding loss, and predicted indices (if requested).
        """
        q_b, q_t, diff, ind = self.encode(input)
        dec = self.decode(q_b, q_t)
        if return_only_recon:
            return dec
        elif return_pred_indices:
            return dec, diff, ind
        else:
            return dec, diff

    def training_step(self, batch, batch_idx):
        """
        Training step for the latent VQ-VAE-2.
        Args:
            batch (torch.Tensor): Input batch tensor.
            batch_idx (int): Index of the current batch.
        Returns:
            torch.Tensor: Total loss for the current batch.
        """
        # -- 1. Prepare data -------------------------------------- #
        inputs = self.get_input(batch)
        recons, qloss, ind = self(inputs, return_pred_indices=True)

        # Decode to pixel‑space once if any perceptual / GAN losses are active
        if isinstance(self.loss, VQLPIPSWithDiscriminator):
            with torch.no_grad():
                inputs_img = self.latents_to_images(inputs)
                recons_img = self.latents_to_images(recons)

        # -- 2. Compute losses ------------------------------------ #
        if isinstance(self.loss, VQLPIPSWithDiscriminator):
            # Two optimizers: (0) generator, (1) discriminator
            opt_g, opt_d = self.optimizers()

            # Generator / VQ-VAE update (optimizer_idx = 0)
            loss_g, log_g = self.loss(
                inputs, recons,
                inputs_img, recons_img,
                qloss,
                optimizer_idx=0,
                global_step=self.global_step,
                predicted_indices=ind,
                split="train"
            )
            opt_g.zero_grad()
            self.manual_backward(loss_g)
            opt_g.step()

            # Discriminator update (optimizer_idx = 1)
            loss_d, log_d = self.loss(
                inputs.detach(), recons.detach(),
                inputs_img.detach(), recons_img.detach(),
                qloss.detach(),
                optimizer_idx=1,
                global_step=self.global_step,
                split="train"
            )
            opt_d.zero_grad()
            self.manual_backward(loss_d)
            opt_d.step()

            total_loss = loss_g + loss_d
            log_dict = {**log_g, **log_d}

        elif isinstance(self.loss, SimpleVQVAELoss):
            total_loss, log_dict = self.loss(
                inputs=inputs,
                recons=recons,
                qloss=qloss,
                split="train",
            )

        # -- 3. Log ------------------------------------------------------------- #
        self.log_dict(log_dict, prog_bar=False, on_step=True, on_epoch=False, sync_dist=True)
        self.log("train/total_loss", total_loss, prog_bar=False,
                 on_step=True, on_epoch=True, sync_dist=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the latent VQ-VAE-2.
        Args:
            batch (torch.Tensor): Input batch tensor.
            batch_idx (int): Index of the current batch.
        Returns:
            torch.Tensor: Total loss for the current batch.
        """
        # -- 1. Prepare data -------------------------------------- #
        inputs = self.get_input(batch)
        recons, qloss, ind = self(inputs, return_pred_indices=True)

        # Pixel‑space reconstructions needed for perceptual / FID / spectral
        if isinstance(self.loss, VQLPIPSWithDiscriminator) or self.track_fid or self.track_spectral:
            with torch.no_grad():
                inputs_img = self.latents_to_images(inputs)
                recons_img = self.latents_to_images(recons)
            if self.track_fid or self.track_spectral:
                self._val_recons_img.append(recons_img.cpu())

        # -- 2. Compute losses ------------------------------------ #
        if isinstance(self.loss, VQLPIPSWithDiscriminator):
            loss_g, log_g = self.loss(
                inputs, recons,
                inputs_img, recons_img,
                qloss,
                optimizer_idx=0,
                global_step=self.global_step,
                predicted_indices=ind,
                split="val"
            )
            loss_d, log_d = self.loss(
                inputs.detach(), recons.detach(),
                inputs_img.detach(), recons_img.detach(),
                qloss.detach(),
                optimizer_idx=1,
                global_step=self.global_step,
                split="val"
            )
            total_loss = loss_g + loss_d
            log_dict   = {**log_g, **log_d}

        elif isinstance(self.loss, SimpleVQVAELoss):
            total_loss, log_dict = self.loss(
                inputs=inputs,
                recons=recons,
                qloss=qloss,
                split="val",
            )

        # -- 3. Log ------------------------------------------------------------- #
        self.log_dict(log_dict, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/total_loss", total_loss, prog_bar=True,
                 on_step=False, on_epoch=True, sync_dist=True)

        return total_loss

    def configure_optimizers(self):
        """ Re-uses the same optimizer strategy as LatentVQVAE """
        if isinstance(self.loss, VQLPIPSWithDiscriminator):
            d_params = list(self.loss.discriminator.parameters())
            d_ids    = set(id(p) for p in d_params)
            g_params = [p for p in self.parameters() if id(p) not in d_ids]

            lr_g = self.learning_rate
            lr_d = 4 * lr_g

            opt_g = torch.optim.Adam(g_params, lr=lr_g, betas=(0.5, 0.999))
            opt_d = torch.optim.Adam(d_params, lr=lr_d, betas=(0.0, 0.999))
            return [opt_g, opt_d]

        elif isinstance(self.loss, SimpleVQVAELoss):
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class LatentAutoencoder(LatentModel):
    def __init__(
            self,
            ddconfig,
            lossconfig,
            learning_rate=1e-4,
            sd_vae_path=None,
            ckpt_path=None,
            ignore_keys=[],
            fid_instance=None,
            spectral_instance=None,
        ):
        """
        LatentAutoencoder to process latents instead of images.
        Args:
            ddconfig: Configuration for the encoder and decoder.
            lossconfig: Configuration for the loss function.
            learning_rate: Learning rate for the optimizer.
            sd_vae_path: Path to the Stable Diffusion VAE model for perceptual loss.
            ckpt_path: Path to a checkpoint to load weights from.
            ignore_keys: Keys to ignore when loading weights from the checkpoint.
            fid_instance: Instance of the FID metric for evaluation.
            spectral_instance: Instance of the Spectral metric for evaluation.
        """
        # Setup general stuff for latent models
        super().__init__(
            ddconfig=ddconfig,
            lossconfig=lossconfig,
            learning_rate=learning_rate,
            sd_vae_path=sd_vae_path,
            fid_instance=fid_instance,
            spectral_instance=spectral_instance,
        )

        # Ensure it is an Autoencoder (not a VAE)
        assert not ddconfig["double_z"]

        # Create encoder/decoder
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)

        # Get parameters from the config
        embed_dim = ddconfig["embed_dim"]
        resolution = ddconfig["resolution"]
        ch_mult = ddconfig["ch_mult"]
        z_channels = ddconfig["z_channels"]

        # Calculate bottleneck spatial dimensions
        bottleneck_resolution = resolution // (2 ** (len(ch_mult) - 1))
        spatial_size = bottleneck_resolution * bottleneck_resolution

        # Single set of features for 1D latent
        self.quant_conv = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(z_channels * spatial_size, embed_dim)
        )

        # De-project from 1D latent
        self.post_quant_conv = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, z_channels * spatial_size),
            torch.nn.Unflatten(1, (z_channels, bottleneck_resolution, bottleneck_resolution))
        )

        # Initialize from checkpoint if provided
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def encode(self, x):
        """
        Encode a latent tensor to a lower dimension.
        Args:
            x (torch.Tensor): Input latent tensor of shape (B, C, H, W).
        Returns:
            torch.Tensor: Encoded tensor of shape (B, embed_dim).
        """
        h = self.encoder(x)
        z = self.quant_conv(h)
        return z

    def decode(self, z):
        """
        Decode a latent tensor to a higher dimension.
        Args:
            z (torch.Tensor): Input latent tensor of shape (B, embed_dim).
        Returns:
            torch.Tensor: Reconstructed tensor of shape (B, C, H, W).
        """
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, x, return_only_recon=False):
        """
        Encode and decode a latent tensor.
        Args:
            x (torch.Tensor): Input latent tensor of shape (B, C, H, W).
            return_only_recon (bool): Whether to return only the reconstructed image.
                Note: This is not needed for the autoencoder, but kept for consistency.
        Returns:
            torch.Tensor: Reconstructed tensor of shape (B, C, H, W).
        """
        z = self.encode(x)
        return self.decode(z)

    def training_step(self, batch, batch_idx):
        """
        Training step for the latent autoencoder.
        Args:
            batch (torch.Tensor): Input batch tensor.
            batch_idx (int): Index of the current batch.
        Returns:
            torch.Tensor: Total loss for the current batch.
        """
        # -- 1. Get data ------------------------------------------ #
        # Get inputs & reconstructions
        inputs = self.get_input(batch)
        recons = self(inputs)

        # Decode latents to images for perceptual loss
        if isinstance(self.loss, AutoencoderLPIPSWithDiscriminator):
            with torch.no_grad():
                inputs_img = self.latents_to_images(inputs)
                recons_img = self.latents_to_images(recons)

        # -- 2. Run training step based on the loss function ------ #

        if isinstance(self.loss, AutoencoderLPIPSWithDiscriminator):
            # -- 2A. LPIPS with Discriminator loss ---------------- #

            # Grab the two optimizers
            opt_g, opt_d = self.optimizers()

            # Generator / VAE update (optimizer_idx = 0)
            loss_g, log_g = self.loss(
                inputs, recons,
                inputs_img, recons_img,
                optimizer_idx=0,
                global_step=self.global_step,
                split="train"
            )

            opt_g.zero_grad()
            self.manual_backward(loss_g)
            opt_g.step()

            # Discriminator update (optimizer_idx = 1)
            # detach everything that should not receive grads
            loss_d, log_d = self.loss(
                inputs.detach(), recons.detach(),
                inputs_img.detach(), recons_img.detach(),
                optimizer_idx=1,
                global_step=self.global_step,
                split="train"
            )

            opt_d.zero_grad()
            self.manual_backward(loss_d)
            opt_d.step()

            # Summarize losses
            log_dict = {**log_g, **log_d}
            total_loss = loss_g + loss_d

        elif isinstance(self.loss, (SimpleAutoencoderLoss)):
            # -- 2B. Simple Autoencoder Loss ---------------------- #

            # Compute loss
            total_loss, log_dict = self.loss(
                inputs=inputs,
                recons=recons,
                split="train"
            )

        # -- 3. Logging ------------------------------------------- #
        self.log_dict(log_dict, prog_bar=False, on_step=True, on_epoch=False, sync_dist=True)
        self.log("train/total_loss", total_loss, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step for the latent autoencoder.
        Args:
            batch (torch.Tensor): Input batch tensor.
            batch_idx (int): Index of the current batch.
        Returns:
            torch.Tensor: Total loss for the current batch.
        """
        # -- 1. Get data ------------------------------------------ #
        # Get inputs & reconstructions
        inputs = self.get_input(batch)
        recons = self(inputs)

        # Decode latents to images for perceptual loss
        if isinstance(self.loss, AutoencoderLPIPSWithDiscriminator) or self.track_fid or self.track_spectral:
            with torch.no_grad():
                inputs_img = self.latents_to_images(inputs)
                recons_img = self.latents_to_images(recons)
            # Store reconstructed images for FID or Spectral calculation
            if self.track_fid or self.track_spectral:
                self._val_recons_img.append(recons_img.cpu())

        # -- 2. Run training step based on the loss function ------ #

        if isinstance(self.loss, AutoencoderLPIPSWithDiscriminator):
            # -- 2A. LPIPS with Discriminator loss ---------------- #

            # Generator / VAE loss (optimizer_idx = 0)
            loss_g, log_g = self.loss(
                inputs, recons,
                inputs_img, recons_img,
                optimizer_idx=0,
                global_step=self.global_step,
                split="val"
            )

            # Discriminator loss (optimizer_idx = 1)
            loss_d, log_d = self.loss(
                inputs.detach(), recons.detach(),
                inputs_img.detach(), recons_img.detach(),
                optimizer_idx=1,
                global_step=self.global_step,
                split="val"
            )

            # Summarize losses
            log_dict = {**log_g, **log_d}
            total_loss = loss_g + loss_d

        elif isinstance(self.loss, SimpleAutoencoderLoss):
            # -- 2B. Simple Autoencoder Loss ---------------------- #

            # Compute loss
            total_loss, log_dict = self.loss(
                inputs=inputs,
                recons=recons,
                split="val"
            )

        # -- 3. Logging ------------------------------------------- #
        self.log_dict(log_dict, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/total_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True)
        return total_loss

    def configure_optimizers(self):
        """
        Configure the optimizers for the model.
        Returns:
            list: List of optimizers.
        """
        if isinstance(self.loss, (AutoencoderLPIPSWithDiscriminator)):
            # Get relevant parameters for VAE and discriminator
            d_params = list(self.loss.discriminator.parameters())
            d_ids = set(id(p) for p in d_params)
            g_params = [p for p in self.parameters() if id(p) not in d_ids]

            # Learning rates
            lr_g = self.learning_rate
            lr_d = 4 * lr_g

            # Use two optimizers for VAE and discriminator
            # First optimizer (generator / VAE)
            opt_g = torch.optim.Adam(
                g_params,
                lr=lr_g, 
                betas=(0.5, 0.999)
            )

            # Second optimizer (discriminator)
            opt_d = torch.optim.Adam(
                d_params,
                lr=lr_d, 
                betas=(0.0, 0.999)
            )

            # Return both optimizers
            return [opt_g, opt_d]
        
        elif isinstance(self.loss, SimpleAutoencoderLoss):
            # Use a single optimizer for Autoencoder
            return torch.optim.Adam(
                self.parameters(), 
                lr=self.learning_rate
            )


class LatentLinearAE(LatentModel):
    """
    Linear Autoencoder for processing latents.
    """
    def __init__(
            self,
            ddconfig,
            lossconfig,
            learning_rate=1e-4,
            sd_vae_path=None,
            ckpt_path=None,
            ignore_keys=[],
            fid_instance=None,
            spectral_instance=None,
        ):
        """
        LatentLinearAE to process latents instead of images.
        Args:
            ddconfig: Configuration for the encoder and decoder.
            lossconfig: Configuration for the loss function.
            learning_rate: Learning rate for the optimizer.
            sd_vae_path: Path to the Stable Diffusion VAE model for perceptual loss.
            ckpt_path: Path to a checkpoint to load weights from.
            ignore_keys: Keys to ignore when loading weights from the checkpoint.
            fid_instance: Instance of the FID metric for evaluation.
            spectral_instance: Instance of the Spectral metric for evaluation.
        """
        super().__init__(
            ddconfig=ddconfig,
            lossconfig=lossconfig,
            learning_rate=learning_rate,
            sd_vae_path=sd_vae_path,
            fid_instance=fid_instance,
            spectral_instance=spectral_instance,
        )

        # Get parameters from the config
        input_dim = ddconfig["input_dim"]
        latent_dim = ddconfig["latent_dim"]

        self.encoder = nn.Linear(input_dim, latent_dim, bias=True)
        self.decoder = nn.Linear(latent_dim, input_dim, bias=True)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def encode(self, x):
        """
        Encode a latent tensor to a lower dimension.
        Args:
            x (torch.Tensor): Input latent tensor of shape (B, C, H, W).
        Returns:
            torch.Tensor: Encoded tensor of shape (B, latent_dim).
        """
        return self.encoder(self._flatten(x))

    def decode(self, z):
        """
        Decode a latent tensor to a higher dimension.
        Args:
            z (torch.Tensor): Input latent tensor of shape (B, latent_dim).
        Returns:
            torch.Tensor: Reconstructed tensor of shape (B, C, H, W).
        """
        return self._unflatten(self.decoder(z))

    def _flatten(self, x):
        """
        Flattens a 4D tensor to a 1D vector.
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
        Returns:
            torch.Tensor: Flattened tensor of shape (B, C*H*W).
        """
        # Store original shape for unflattening
        if not hasattr(self, "orig_shape"):
            self.orig_shape = x.shape[1:]
        return x.flatten(start_dim=1)

    def _unflatten(self, z):
        """
        Unflattens a 1D vector to a 4D tensor.
        Args:
            z (torch.Tensor): Input tensor of shape (B, latent_dim).
        Returns:
            torch.Tensor: Unflattened tensor of shape (B, C, H, W).
        """
        if not hasattr(self, "orig_shape"):
            raise RuntimeError("Original shape not stored. Call _flatten() first.")
        return z.view(z.shape[0], self.orig_shape[0], self.orig_shape[1], self.orig_shape[2])

    def forward(self, x, return_only_recon=False):
        """
        Encode and decode a latent tensor.
        Args:
            x (torch.Tensor): Input latent tensor of shape (B, C, H, W).
            return_only_recon (bool): Whether to return only the reconstructed image.
                Note: This is not needed for the autoencoder, but kept for consistency.
        Returns:
            torch.Tensor: Reconstructed tensor of shape (B, C, H, W).
        """
        z = self.encoder(self._flatten(x))
        x_hat = self._unflatten(self.decoder(z))
        return x_hat

    def training_step(self, batch, batch_idx):
        """
        Training step for the linear autoencoder.
        Args:
            batch (torch.Tensor): Input batch tensor.
            batch_idx (int): Index of the current batch.
        Returns:
            torch.Tensor: Total loss for the current batch.
        """
        # -- 1. Get data ------------------------------------------ #
        # Get inputs & reconstructions
        inputs = self.get_input(batch)
        recons = self(inputs)

        # Decode latents to images for perceptual loss
        if isinstance(self.loss, AutoencoderLPIPSWithDiscriminator):
            with torch.no_grad():
                inputs_img = self.latents_to_images(inputs)
                recons_img = self.latents_to_images(recons)

        # -- 2. Run training step based on the loss function ------ #

        if isinstance(self.loss, AutoencoderLPIPSWithDiscriminator):
            # -- 2A. LPIPS with Discriminator loss ---------------- #

            # Grab the two optimizers
            opt_g, opt_d = self.optimizers()

            # Generator / VAE update (optimizer_idx = 0)
            loss_g, log_g = self.loss(
                inputs, recons,
                inputs_img, recons_img,
                optimizer_idx=0,
                global_step=self.global_step,
                split="train"
            )

            opt_g.zero_grad()
            self.manual_backward(loss_g)
            opt_g.step()

            # Discriminator update (optimizer_idx = 1)
            # detach everything that should not receive grads
            loss_d, log_d = self.loss(
                inputs.detach(), recons.detach(),
                inputs_img.detach(), recons_img.detach(),
                optimizer_idx=1,
                global_step=self.global_step,
                split="train"
            )

            opt_d.zero_grad()
            self.manual_backward(loss_d)
            opt_d.step()

            # Summarize losses
            log_dict = {**log_g, **log_d}
            total_loss = loss_g + loss_d

        elif isinstance(self.loss, LPIPSMSELoss):
            total_loss, log_dict = self.loss(
                inputs, recons,
                inputs_img, recons_img,
                split="train"
            )

        # -- 3. Logging ------------------------------------------- #
        self.log_dict(log_dict, prog_bar=False, on_step=True, on_epoch=False, sync_dist=True)
        self.log("train/total_loss", total_loss, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the linear autoencoder.
        Args:
            batch (torch.Tensor): Input batch tensor.
            batch_idx (int): Index of the current batch.
        Returns:
            torch.Tensor: Total loss for the current batch.
        """
        # -- 1. Get data ------------------------------------------ #
        # Get inputs & reconstructions
        inputs = self.get_input(batch)
        recons = self(inputs)

        # Decode latents to images for perceptual loss
        if isinstance(self.loss, AutoencoderLPIPSWithDiscriminator) or self.track_fid or self.track_spectral:
            with torch.no_grad():
                inputs_img = self.latents_to_images(inputs)
                recons_img = self.latents_to_images(recons)
            # Store reconstructed images for FID or Spectral calculation
            if self.track_fid or self.track_spectral:
                self._val_recons_img.append(recons_img.cpu())

        # -- 2. Run training step based on the loss function ------ #
        if isinstance(self.loss, AutoencoderLPIPSWithDiscriminator):
            # -- 2A. LPIPS with Discriminator loss ---------------- #

            # Generator / VAE loss (optimizer_idx = 0)
            loss_g, log_g = self.loss(
                inputs, recons,
                inputs_img, recons_img,
                optimizer_idx=0,
                global_step=self.global_step,
                split="val"
            )

            # Discriminator loss (optimizer_idx = 1)
            loss_d, log_d = self.loss(
                inputs.detach(), recons.detach(),
                inputs_img.detach(), recons_img.detach(),
                optimizer_idx=1,
                global_step=self.global_step,
                split="val"
            )

            # Summarize losses
            log_dict = {**log_g, **log_d}
            total_loss = loss_g + loss_d

        elif isinstance(self.loss, LPIPSMSELoss):
            total_loss, log_dict = self.loss(
                inputs, recons,
                inputs_img, recons_img,
                split="val"
            )

        # -- 3. Logging ------------------------------------------- #
        self.log_dict(log_dict, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/total_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return total_loss

    def configure_optimizers(self):
        """
        Configure the optimizers for the model.
        Returns:
            list: List of optimizers.
        """
        if isinstance(self.loss, (AutoencoderLPIPSWithDiscriminator)):
            # Get relevant parameters for VAE and discriminator
            d_params = list(self.loss.discriminator.parameters())
            d_ids = set(id(p) for p in d_params)
            g_params = [p for p in self.parameters() if id(p) not in d_ids]

            # Learning rates
            lr_g = self.learning_rate
            lr_d = 4 * lr_g

            # Use two optimizers for VAE and discriminator
            # First optimizer (generator / VAE)
            opt_g = torch.optim.Adam(
                g_params,
                lr=lr_g, 
                betas=(0.5, 0.999)
            )

            # Second optimizer (discriminator)
            opt_d = torch.optim.Adam(
                d_params,
                lr=lr_d, 
                betas=(0.0, 0.999)
            )

            # Return both optimizers
            return [opt_g, opt_d]
        
        elif isinstance(self.loss, LPIPSMSELoss):
            # Use a single optimizer for Autoencoder
            return torch.optim.Adam(
                self.parameters(), 
                lr=self.learning_rate
            )