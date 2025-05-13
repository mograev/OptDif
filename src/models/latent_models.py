from contextlib import contextmanager
from tqdm import tqdm

import torch
from torch import nn
import torch.distributed as dist
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl
from torchvision.utils import make_grid

# Import Stable Diffusion VAE decoder for optional pixel-space losses
from diffusers import AutoencoderKL

from src.models.modules.autoencoder import Encoder, Decoder
from src.models.modules.distributions import DiagonalGaussianDistribution
from src.models.modules.quantize import VectorQuantizer2 as VectorQuantizer
from src.models.modules.utils import instantiate_from_config
from src.models.modules.losses import SimpleVAELoss, VAEWithDiscriminator, LPIPSWithDiscriminator, VQLPIPSWithDiscriminator


class LatentVAE(pl.LightningModule):
    def __init__(self,
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
        super().__init__()
        
        # Create encoder/decoder for variational autoencoder
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        
        # Ensure it is a VAE
        assert ddconfig["double_z"]

        # Get embedding dimension from the config
        embed_dim = ddconfig['embed_dim']

        # Calculate bottleneck spatial dimensions
        bottleneck_resolution = ddconfig["resolution"] // (2 ** (len(ddconfig["ch_mult"])-1))
        spatial_size = bottleneck_resolution * bottleneck_resolution

        # Create custom layers for flattening to 1D latent space
        self.quant_conv = torch.nn.Sequential(
            torch.nn.Flatten(),  # Flatten spatial dimensions to 1D
            torch.nn.Linear(2*ddconfig["z_channels"] * spatial_size, 2*embed_dim)  # Project to flat latent dim (2* for mean/logvar)
        )
        
        # Create custom layers for reshaping from 1D latent back to spatial
        self.post_quant_conv = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, ddconfig["z_channels"] * spatial_size),  # Project from flat latent to spatial
            torch.nn.Unflatten(1, (ddconfig["z_channels"], bottleneck_resolution, bottleneck_resolution))  # Reshape to spatial
        )

        # Loss setup
        self.learning_rate = learning_rate
        self.loss = instantiate_from_config(lossconfig)

        # Stable diffusion VAE for perceptual loss
        if sd_vae_path is not None:
            # Store path in hyperparameters
            self.sd_vae_path = sd_vae_path
            # Load weights from a pre-trained Stable Diffusion VAE
            self.sd_vae = AutoencoderKL.from_pretrained(sd_vae_path, subfolder="vae")
            self.sd_vae.eval()
            self.sd_vae.requires_grad_(False)

        # Manual optimization if Discriminator is used in loss function
        if isinstance(self.loss, (VAEWithDiscriminator, LPIPSWithDiscriminator)):
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
            "ignore_keys",
            "learning_rate",
        )

        # Initialize from checkpoint if provided
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

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
    
    def forward(self, input, sample_posterior=True):
        """
        Encode and decode a latent tensor.
        Args:
            input (torch.Tensor): Input latent tensor of shape (B, C, H, W).
            sample_posterior (bool): Whether to sample from the posterior distribution.
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
        return dec, posterior
    
    def get_input(self, batch):
        """
        Get the input tensor from the batch and ensure its correct dtype.
        Args:
            batch (torch.Tensor): Input batch tensor.
        Returns:
            torch.Tensor: Input tensor with correct dtype.
        """
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

        # Get the last layer of the decoder for adaptive loss
        last_layer = self.get_last_layer()

        # Decode latents to images for perceptual loss
        if isinstance(self.loss, (LPIPSWithDiscriminator, VAEWithDiscriminator)):
            with torch.no_grad():
                inputs_img = self.latents_to_images(inputs)
                recons_img = self.latents_to_images(recons)

        # -- 2. Run training step based on the loss function ------ #

        if isinstance(self.loss, LPIPSWithDiscriminator):
            # -- 2A. LPIPS with Discriminator loss ---------------- #

            # Grab the two optimizers
            opt_vae, opt_disc = self.optimizers()

            # Generator / VAE update (optimizer_idx = 0)
            loss_gen, log_gen = self.loss(
                inputs, recons,
                inputs_img, recons_img,
                posterior,
                optimizer_idx=0,
                global_step=self.global_step,
                split="train"
            )

            opt_vae.zero_grad()
            self.manual_backward(loss_gen)
            opt_vae.step()

            # Discriminator update (optimizer_idx = 1)
            # detach everything that should not receive grads
            loss_disc, log_disc = self.loss(
                inputs.detach(), recons.detach(),
                inputs_img.detach(), recons_img.detach(),
                posterior.detach(),
                optimizer_idx=1,
                global_step=self.global_step,
                split="train"
            )

            opt_disc.zero_grad()
            self.manual_backward(loss_disc)
            opt_disc.step()

            # Summarize losses
            log_dict = {**log_gen, **log_disc}
            total_loss = loss_gen + loss_disc

        elif isinstance(self.loss, VAEWithDiscriminator):
            # -- 2B. VAE with Discriminator loss ------------------ #

            # Grab the two optimizers
            opt_vae, opt_disc = self.optimizers()

            # Generator / VAE update (optimizer_idx = 0)
            loss_gen, log_dict_gen = self.loss(
                inputs=inputs,
                recons=recons,
                posterior=posterior,
                optimizer_idx=0,
                global_step=self.global_step,
                split="train"
            )

            opt_vae.zero_grad()
            self.manual_backward(loss_gen)  
            opt_vae.step()

            # Discriminator update (optimizer_idx = 1)
            loss_disc, log_dict_disc = self.loss(
                inputs=inputs.detach(),
                recons=recons.detach(),
                posterior=posterior.detach(),
                optimizer_idx=1,
                global_step=self.global_step,
                split="train"
            )
            opt_disc.zero_grad()
            self.manual_backward(loss_disc)
            opt_disc.step()

            # Summarize losses
            log_dict = {**log_dict_gen, **log_dict_disc}
            total_loss = loss_gen + loss_disc
            
        else:
            # -- 2C. Automatic optimization (no discriminator) ---- #

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

        # Get the last layer of the decoder for adaptive loss
        last_layer = self.get_last_layer()

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
            loss_gen, log_gen = self.loss(
                inputs, recons,
                inputs_img, recons_img,
                posterior,
                optimizer_idx=0,
                global_step=self.global_step,
                split="val"
            )

            # Discriminator loss (optimizer_idx = 1)
            loss_disc, log_disc = self.loss(
                inputs.detach(), recons.detach(),
                inputs_img.detach(), recons_img.detach(),
                posterior.detach(),
                optimizer_idx=1,
                global_step=self.global_step,
                split="val"
            )

            # Summarize losses
            log_dict = {**log_gen, **log_disc}
            total_loss = loss_gen + loss_disc

        elif isinstance(self.loss, VAEWithDiscriminator):
            # -- 2B. VAE with Discriminator loss ------------------ #

            # Generator / VAE update (optimizer_idx = 0)
            loss_gen, log_dict_gen = self.loss(
                inputs=inputs,
                recons=recons,
                posterior=posterior,
                optimizer_idx=0,
                global_step=self.global_step,
                split="val"
            )

            # Discriminator update (optimizer_idx = 1)
            loss_disc, log_dict_disc = self.loss(
                inputs=inputs.detach(),
                recons=recons.detach(),
                posterior=posterior.detach(),
                optimizer_idx=1,
                global_step=self.global_step,
                split="val"
            )

            # Summarize losses
            log_dict = {**log_dict_gen, **log_dict_disc}
            total_loss = loss_gen + loss_disc
            
        else:
            # -- 2C. Automatic optimization (no discriminator) ---- #

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
            disc_params = list(self.loss.discriminator.parameters())
            disc_ids = set(id(p) for p in disc_params)
            vae_params = [p for p in self.parameters() if id(p) not in disc_ids]

            # Learning rates
            lr_vae = self.learning_rate
            lr_disc = 4 * lr_vae

            # Use two optimizers for VAE and discriminator
            # First optimizer (generator / VAE)
            opt_vae = torch.optim.Adam(
                vae_params,
                lr=lr_vae, 
                betas=(0.5, 0.999)
            )

            # Second optimizer (discriminator)
            opt_disc = torch.optim.Adam(
                disc_params,
                lr=lr_disc, 
                betas=(0.0, 0.999)
            )

            # Return both optimizers
            return [opt_vae, opt_disc]

        else:
            # Use a single optimizer for VAE
            return torch.optim.Adam(
                self.parameters(), 
                lr=self.learning_rate
            )
    
    def get_last_layer(self):
        """
        Get the last layer of the decoder for adaptive loss.
        Returns:
            torch.nn.Module: Last layer of the decoder.
        """
        return self.decoder.conv_out if hasattr(self.decoder, 'conv_out') else None
    
    def on_train_start(self):
        """
        Grab a fixed mini-batch of latents for logging.
        """
        if self.global_rank == 0:
            # Grab the first validation batch and keep up to 16 examples
            val_batch = next(iter(self.trainer.datamodule.val_dataloader()))
            latents = val_batch[0] if isinstance(val_batch, (list, tuple)) else val_batch
            # Preserve batch dimension (B,C,H,W) and move to the right device
            self.fixed_latents = latents[:16].to(self.device)

    @torch.no_grad()
    def _log_reconstructions(self):
        """
        Log reconstructed images for visualization.
        """
        # Ensure tensor is on the correct device and has a batch dimension
        inputs = self.fixed_latents.to(self.device)
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(0)
        recons = self(inputs)[0]

        # Decode latents to images
        inputs_img = self.latents_to_images(inputs)
        recons_img = self.latents_to_images(recons)

        # Create grid of images
        grid = make_grid(torch.cat([inputs_img, recons_img], dim=0), nrow=8, normalize=True)

        # Log the grid of images
        self.logger.experiment.add_image("reconstructions", grid, self.global_step)

    def on_validation_epoch_start(self):
        """
        Initialize list to store reconstructed images for metric calculation.
        """
        if self.track_fid or self.track_spectral:
            # Initialize list to store reconstructed images for metric calculation
            self._val_recons_img = []

        return super().on_validation_epoch_start()

    def on_validation_epoch_end(self):
        """
        Log validation metrics and calculate FID/Spectral scores if applicable.
        """
        # Skip logging if sanity checking
        if self.trainer.sanity_checking:
            return super().on_validation_epoch_end()
        
        # -- Validation loss -------------------------------------- #
        if self.global_rank == 0:
            print(f"Epoch {self.current_epoch}:")
            print(f"~ Validation loss: {self.trainer.callback_metrics['val/total_loss']}")

        # -- Get reconstructed images ----------------------------- #
        if self.track_fid or self.track_spectral:
            # Concatenate all batches of reconstructed images
            recon_img = torch.cat(self._val_recons_img, dim=0)

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



class LatentVQVAE(pl.LightningModule):
    def __init__(
            self,
            ddconfig,
            lossconfig,
            learning_rate=1e-4,
            sd_vae_path=None,
            ckpt_path=None,
            ignore_keys=[],
            remap=None,
            sane_index_shape=False,
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
            remap: Remapping for quantization.
            sane_index_shape: Whether to use a sane index shape for quantization.
            fid_instance: Instance of the FID metric for evaluation.
            spectral_instance: Instance of the Spectral metric for evaluation.
        """
        super().__init__()

        # Create encoder/decoder for VQVAE
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)

        # Setup quantizer
        self.quantize = VectorQuantizer(ddconfig["n_embed"] ,ddconfig["embed_dim"], beta=0.05,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], ddconfig["embed_dim"], 1)
        self.post_quant_conv = torch.nn.Conv2d(ddconfig["embed_dim"], ddconfig["z_channels"], 1)

        # Loss setup
        self.learning_rate = learning_rate
        self.loss = instantiate_from_config(lossconfig)

        # Stable diffusion VAE for perceptual loss
        if sd_vae_path is not None:
            # Store path in hyperparameters
            self.sd_vae_path = sd_vae_path
            # Load weights from a pre-trained Stable Diffusion VAE
            self.sd_vae = AutoencoderKL.from_pretrained(sd_vae_path, subfolder="vae")
            self.sd_vae.eval()
            self.sd_vae.requires_grad_(False)

        # Manual optimization if Discriminator is used in loss function
        if isinstance(self.loss, (VQLPIPSWithDiscriminator)):
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
            "ignore_keys",
            "learning_rate",
            "remap",
            "sane_index_shape", 
        )
        
        # Initialize from checkpoint if provided
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

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
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        """
        Encode a latent tensor to a lower dimension.
        Args:
            x (torch.Tensor): Input latent tensor of shape (B, C, H, W).
        Returns:
            tuple: Quantized tensor, embedding loss, and quantization info.
        """
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

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
            quant (torch.Tensor): Input quantized tensor of shape (B, C, H, W).
        Returns:
            torch.Tensor: Reconstructed tensor of shape (B, C, H, W).
        """
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        """
        Decode an unquantized tensor to a higher dimension.
        Args:
            code_b (torch.Tensor): Input unquantized tensor of shape (B, C, H, W).
        Returns:
            torch.Tensor: Reconstructed tensor of shape (B, C, H, W).
        """
        quant_b = self.quantize(code_b)[0]
        dec = self.decode(quant_b)
        return dec
    
    def forward(self, input, return_pred_indices=False):
        """
        Encode and decode a latent tensor.
        Args:
            input (torch.Tensor): Input latent tensor of shape (B, C, H, W).
            return_pred_indices (bool): Whether to return the predicted indices.
        Returns:
            tuple: Reconstructed tensor, embedding loss, and predicted indices (if requested).
        """
        quant, diff, (_,_,ind) = self.encode(input)
        dec = self.decode(quant)
        if return_pred_indices:
            return dec, diff, ind
        return dec, diff

    def get_input(self, batch):
        """
        Get the input tensor from the batch and ensure its correct dtype.
        Args:
            batch (torch.Tensor): Input batch tensor.
        Returns:
            torch.Tensor: Input tensor with correct dtype.
        """
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
        return images

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

        # Get the last layer of the decoder for adaptive loss
        last_layer = self.get_last_layer()

        # Decode latents to images for perceptual loss
        if isinstance(self.loss, VQLPIPSWithDiscriminator):
            with torch.no_grad():
                inputs_img = self.latents_to_images(inputs)
                recons_img = self.latents_to_images(recons)

        # -- 2. Run training step based on the loss function ------ #

        if isinstance(self.loss, VQLPIPSWithDiscriminator):
            # -- 2A. LPIPS with Discriminator loss ---------------- #

            # Grab the two optimizers
            opt_vae, opt_disc = self.optimizers()

            # Generator / VAE update (optimizer_idx = 0)
            loss_gen, log_gen = self.loss(
                inputs, recons,
                inputs_img, recons_img,
                qloss,
                optimizer_idx=0,
                global_step=self.global_step,
                predicted_indices=ind,
                split="train"
            )

            opt_vae.zero_grad()
            self.manual_backward(loss_gen)
            opt_vae.step()

            # Discriminator update (optimizer_idx = 1)
            # detach everything that should not receive grads
            loss_disc, log_disc = self.loss(
                inputs.detach(), recons.detach(),
                inputs_img.detach(), recons_img.detach(),
                qloss.detach(),
                optimizer_idx=1,
                global_step=self.global_step,
                split="train"
            )

            opt_disc.zero_grad()
            self.manual_backward(loss_disc)
            opt_disc.step()

            # Summarize losses
            log_dict = {**log_gen, **log_disc}
            total_loss = loss_gen + loss_disc

        else:
            # -- 2B. Automatic optimization (no discriminator) ---- #

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

        # Get the last layer of the decoder for adaptive loss
        last_layer = self.get_last_layer()

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
            loss_gen, log_gen = self.loss(
                inputs, recons,
                inputs_img, recons_img,
                qloss,
                optimizer_idx=0,
                global_step=self.global_step,
                predicted_indices=ind,
                split="val"
            )

            # Discriminator loss (optimizer_idx = 1)
            loss_disc, log_disc = self.loss(
                inputs.detach(), recons.detach(),
                inputs_img.detach(), recons_img.detach(),
                qloss.detach(),
                optimizer_idx=1,
                global_step=self.global_step,
                split="val"
            )

            # Summarize losses
            log_dict = {**log_gen, **log_disc}
            total_loss = loss_gen + loss_disc

        else:
            # -- 2B. Automatic optimization (no discriminator) ---- #

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
            disc_params = list(self.loss.discriminator.parameters())
            disc_ids = set(id(p) for p in disc_params)
            vae_params = [p for p in self.parameters() if id(p) not in disc_ids]

            # Learning rates
            lr_vae = self.learning_rate
            lr_disc = 4 * lr_vae

            # Use two optimizers for VAE and discriminator
            # First optimizer (generator / VAE)
            opt_vae = torch.optim.Adam(
                vae_params,
                lr=lr_vae, 
                betas=(0.5, 0.999)
            )

            # Second optimizer (discriminator)
            opt_disc = torch.optim.Adam(
                disc_params,
                lr=lr_disc, 
                betas=(0.0, 0.999)
            )

            # Return both optimizers
            return [opt_vae, opt_disc]

    def get_last_layer(self):
        """
        Get the last layer of the decoder for adaptive loss.
        Returns:
            torch.nn.Module: Last layer of the decoder.
        """
        return self.decoder.conv_out if hasattr(self.decoder, 'conv_out') else None
    
    def on_train_start(self):
        """
        Grab a fixed mini-batch of latents for logging.
        """
        if self.global_rank == 0:
            # Grab the first validation batch and keep up to 16 examples
            val_batch = next(iter(self.trainer.datamodule.val_dataloader()))
            latents = val_batch[0] if isinstance(val_batch, (list, tuple)) else val_batch
            # Preserve batch dimension (B,C,H,W) and move to the right device
            self.fixed_latents = latents[:16].to(self.device)

    @torch.no_grad()
    def _log_reconstructions(self):
        """
        Log reconstructed images for visualization.
        """
        # Ensure tensor is on the correct device and has a batch dimension
        inputs = self.fixed_latents.to(self.device)
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(0)
        recons = self(inputs)[0]

        # Decode latents to images
        inputs_img = self.latents_to_images(inputs)
        recons_img = self.latents_to_images(recons)

        # Create grid of images
        grid = make_grid(torch.cat([inputs_img, recons_img], dim=0), nrow=8, normalize=True)

        # Log the grid of images
        self.logger.experiment.add_image("reconstructions", grid, self.global_step)


    def on_validation_epoch_start(self):
        """
        Initialize list to store reconstructed images for metric calculation.
        """
        if self.track_fid or self.track_spectral:
            # Initialize list to store reconstructed images for metric calculation
            self._val_recons_img = []

        return super().on_validation_epoch_start()
    
    def on_validation_epoch_end(self):
        """
        Log validation metrics and calculate FID/Spectral scores if applicable.
        """
        # Skip logging if sanity checking
        if self.trainer.sanity_checking:
            return super().on_validation_epoch_end()
        
        # -- Validation loss -------------------------------------- #
        if self.global_rank == 0:
            print(f"Epoch {self.current_epoch}:")
            print(f"~ Validation loss: {self.trainer.callback_metrics['val/total_loss']}")

        # -- Get reconstructed images ----------------------------- #
        if self.track_fid or self.track_spectral:
            # Concatenate all batches of reconstructed images
            recon_img = torch.cat(self._val_recons_img, dim=0)

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


class LatentAutoencoder(pl.LightningModule):
    def __init__(
        self,
        ddconfig,
        lossconfig,
        embed_dim,
        ckpt_path=None,
        ignore_keys=[],
        monitor=None
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Create encoder/decoder for non-variational autoencoder
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.automatic_optimization = True  # single optimizer now

        # Ensure it is an Autoencoder (not a VAE)
        assert not ddconfig["double_z"]

        # Calculate spatial dimensions
        bottleneck_resolution = ddconfig["resolution"] // (
            2 ** (len(ddconfig["ch_mult"]) - 1)
        )
        spatial_size = bottleneck_resolution * bottleneck_resolution

        # Single set of features for 1D latent
        self.quant_conv = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(
                ddconfig["z_channels"] * spatial_size,
                embed_dim
            )
        )

        # De-project from 1D latent
        self.post_quant_conv = torch.nn.Sequential(
            torch.nn.Linear(
                embed_dim,
                ddconfig["z_channels"] * spatial_size
            ),
            torch.nn.Unflatten(
                1,
                (ddconfig["z_channels"], bottleneck_resolution, bottleneck_resolution)
            )
        )

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.save_hyperparameters()
        self.learning_rate = 1e-4

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in list(sd.keys()):
            for ik in ignore_keys:
                if k.startswith(ik):
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Loaded state dict from {path}")

    def encode(self, x):
        h = self.encoder(x)
        z = self.quant_conv(h)
        return z

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def get_input(self, batch):
        return batch.to(memory_format=torch.contiguous_format).float()

    def training_step(self, batch, batch_idx):
        inputs = self.get_input(batch)
        reconstructions = self(inputs)
        loss, log_dict = self.loss(inputs=inputs, recons=reconstructions)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch)
        reconstructions = self(inputs)
        val_loss, log_dict = self.loss(inputs=inputs, recons=reconstructions)
        self.log("val_loss", val_loss, prog_bar=True, on_step=False, on_epoch=True)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    




class LatentLinearAE(pl.LightningModule):
    """
    Maps a SD-latent tensor to a lower dimension and back with two
    linear layers.
    """
    def __init__(
            self,
            lossconfig,
            sd_vae_path=None,
            d_in=16384,
            d_mid=512,
            lr=1e-4
        ):
        super().__init__()
        self.lr = lr

        self.encoder = nn.Linear(d_in, d_mid, bias=True)
        self.decoder = nn.Linear(d_mid, d_in, bias=True)

        self.loss = instantiate_from_config(lossconfig)

        if sd_vae_path is not None:
            # Load weights from a pre-trained Stable Diffusion VAE
            self.sd_vae = AutoencoderKL.from_pretrained(sd_vae_path, subfolder="vae")
            self.sd_vae.eval()
            self.sd_vae.requires_grad_(False)
        else:
            self.sd_vae = None

        self.save_hyperparameters()

    def encode(self, x):
        """Encodes a latent tensor to a lower dimension."""
        return self.encoder(self._flatten(x))

    def decode(self, z):
        """Decodes a latent tensor to a higher dimension."""
        return self._unflatten(self.decoder(z))

    def latents_to_images(self, latents):
        """Decode Stable-Diffusion latents to pixel space."""
        if self.sd_vae is None:
            raise RuntimeError("latents_to_images() called but sd_vae is not initialized.")
        # Move decoder to correct device if necessary
        self.sd_vae.to(latents.device)
        images = self.sd_vae.decode(latents).sample

        return images

    def _flatten(self, x):
        """Flattens a latent tensor to a 1D vector."""
        # Store original shape for unflattening
        if not hasattr(self, "orig_shape"):
            self.orig_shape = x.shape[1:]
        return x.flatten(start_dim=1)

    def _unflatten(self, z):
        """Unflattens a 1D vector to the original latent tensor shape."""
        if not hasattr(self, "orig_shape"):
            raise RuntimeError("Original shape not stored. Call _flatten() first.")
        return z.view(z.shape[0], self.orig_shape[0], self.orig_shape[1], self.orig_shape[2])

    def forward(self, x):
        """Encodes and decodes a latent tensor."""
        z = self.encoder(self._flatten(x))
        x_hat = self._unflatten(self.decoder(z))
        return x_hat

    def training_step(self, batch, _):
        """Training step for the linear autoencoder."""
        inputs = batch.float()
        recons = self(inputs)

        # Decoding to pixel space for perceptual loss
        with torch.no_grad():
            inputs_img = self.latents_to_images(inputs)
            recons_img = self.latents_to_images(recons)
        loss, log_dict = self.loss(inputs, recons, inputs_img, recons_img)

        # Free up memory
        del inputs_img, recons_img
        
        # Log metrics
        self.log_dict(log_dict, prog_bar=False, on_step=True, on_epoch=False, sync_dist=True)
        
        return loss

    def validation_step(self, batch, _):
        """Validation step for the linear autoencoder."""
        inputs = batch.float()
        recons = self(inputs)

        # Decoding to pixel space for perceptual loss
        with torch.no_grad():
            inputs_img = self.latents_to_images(inputs)
            recons_img = self.latents_to_images(recons)
        loss, log_dict = self.loss(inputs, recons, inputs_img, recons_img, split="val")

        # Free up memory
        del inputs_img, recons_img

        # Log metrics
        self.log_dict(log_dict, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        """Configures the optimizer for the linear autoencoder."""
        return torch.optim.Adam(self.parameters(), lr=self.lr)
