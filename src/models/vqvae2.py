import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F
import torch.distributed as dist
from torchvision.utils import make_grid

from src.models.modules.autoencoder  import Encoder, Decoder
from src.models.modules.quantize import VectorQuantizer2 as VectorQuantizer
from src.models.modules.losses import VQLPIPSWithDiscriminator, SimpleVQVAELoss
from src.models.modules.utils import instantiate_from_config

class VQVAE2(pl.LightningModule):
    """
    Two-level VQ-VAE-2 that works directly in image space.
    A top-level code captures global structure; bottom refines details.
    """
    def __init__(
        self,
        ddconfig: dict,
        lossconfig: dict,
        learning_rate: float = 1e-4,
        ckpt_path: str | None = None,
        ignore_keys: list = [],
        fid_instance=None,
        spectral_instance=None,
    ):
        """
        Initialize the VQVAE2 model.
        Args:
            ddconfig (dict): Configuration for the encoder/decoder architecture.
            lossconfig (dict): Configuration for the loss function.
            learning_rate (float): Learning rate for the optimizer.
            ckpt_path (str | None): Path to a checkpoint to restore weights from.
            ignore_keys (list): List of keys to ignore when loading weights.
            fid_instance: Instance of FID metric, if any.
            spectral_instance: Instance of spectral metric, if any.
        """
        super().__init__()

        # Store learning rate
        self.learning_rate = learning_rate

        # Loss setup
        self.loss = instantiate_from_config(lossconfig)

        # Manual optimization if Discriminator is used in loss function
        if isinstance(self.loss, VQLPIPSWithDiscriminator):
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
            "ckpt_path",
            "learning_rate",
        )

        # split configs
        assert "bottom" in ddconfig and "top" in ddconfig, \
            "ddconfig must have 'bottom' and 'top' keys"
        bottom_cfg = ddconfig["bottom"]
        top_cfg    = ddconfig["top"]

        # -- Encoders & Decoders ---------------------------------- #
        self.encoder_bottom = Encoder(**bottom_cfg)
        self.encoder_top    = Encoder(**top_cfg)

        # Upsample top→bottom, then final decode
        self.decoder_top    = Decoder(**top_cfg)
        self.decoder_bottom = Decoder(**bottom_cfg)


        # -- Dimensions ------------------------------------------- #
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

    def encode(self, x):
        """
        Encode input image into quantized bottom & top codes.
        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).
        Returns:
            q_b (torch.Tensor): Quantized bottom code of shape (B, Cb, Hb, Wb).
            q_t (torch.Tensor): Quantized top code of shape (B, Ct, Ht, Wt).
            diff_total (torch.Tensor): Total quantization loss
                                       (sum of bottom and top quantization losses).
            ind (torch.Tensor): Concatenated indices of quantized codes.
        """
        # bottom features
        h_b = self.encoder_bottom(x)                     # (B, Cb, Hb, Wb)

        # top features & quantize
        h_t = self.encoder_top(h_b)                      # (B, Ct, Ht, Wt)
        h_t = self.quant_conv_top(h_t)
        q_t, diff_t, (_,_,ind_t) = self.quantize_top(h_t)

        # decode top→bottom res
        d_t = self.decoder_top(self.post_quant_conv_top(q_t))

        # residual bottom quant
        h_b = self.quant_conv_bottom(h_b + d_t)
        q_b, diff_b, (_,_,ind_b) = self.quantize_bottom(h_b)

        # sum losses & concat indices
        diff_total = diff_t + diff_b
        ind = torch.cat([
            ind_t.reshape(ind_t.size(0), -1),
            ind_b.reshape(ind_b.size(0), -1)
        ], dim=1)

        return q_b, q_t, diff_total, ind

    def decode(self, q_b, q_t):
        """
        Reconstruct image from quantized bottom & top codes.
        Args:
            q_b (torch.Tensor): Quantized bottom code of shape (B, Cb, Hb, Wb).
            q_t (torch.Tensor): Quantized top code of shape (B, Ct, Ht, Wt).
        Returns:
            x (torch.Tensor): Reconstructed image tensor of shape (B, C, H, W).
        """
        d_t = self.decoder_top(self.post_quant_conv_top(q_t))
        x   = self.decoder_bottom(self.post_quant_conv_bottom(q_b) + d_t)
        return x

    def decode_code(self, code_b, code_t):
        """
        Decode image from quantized bottom & top codes.
        Args:
            code_b (torch.Tensor): Quantized bottom code of shape (B, Cb, Hb, Wb).
            code_t (torch.Tensor): Quantized top code of shape (B, Ct, Ht, Wt).
        Returns:
            x (torch.Tensor): Reconstructed image tensor of shape (B, C, H, W).
        """
        if code_b.dim()==1:
            code_b = code_b.unsqueeze(0)
        if code_t.dim()==1:
            code_t = code_t.unsqueeze(0)

        q_b = self.quantize_bottom.get_codebook_entry(
            code_b, shape=(code_b.shape[0], *self.latent_dim_bottom)
        )
        q_t = self.quantize_top.get_codebook_entry(
            code_t, shape=(code_t.shape[0], *self.latent_dim_top)
        )
        return self.decode(q_b, q_t)

    def forward(self, x, return_pred_indices=False, return_only_recon=False):
        """
        Forward pass through the model.
        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).
            return_pred_indices (bool): Whether to return predicted indices.
            return_only_recon (bool): Whether to return only the reconstruction.
        Returns:
            dec (torch.Tensor): Reconstructed image tensor of shape (B, C, H, W).
            diff (torch.Tensor): Total quantization loss.
            ind (torch.Tensor, optional): Concatenated indices of quantized codes.
        """
        q_b, q_t, diff, ind = self.encode(x)
        dec = self.decode(q_b, q_t)
        if return_only_recon:
            return dec
        if return_pred_indices:
            return dec, diff, ind
        return dec, diff

    def training_step(self, batch, batch_idx):
        """
        Training step for the VQ-VAE-2.
        Args:
            batch (torch.Tensor): Input batch tensor.
            batch_idx (int): Index of the current batch.
        Returns:
            torch.Tensor: Total loss for the current batch.
        """
        # -- 1. Prepare data -------------------------------------- #
        inputs = self.get_input(batch)
        recons, qloss, ind = self(inputs, return_pred_indices=True)

        # -- 2. Compute losses ------------------------------------ #
        if isinstance(self.loss, VQLPIPSWithDiscriminator):
            # Two optimizers: (0) generator, (1) discriminator
            opt_g, opt_d = self.optimizers()

            # Generator / VQ-VAE update (optimizer_idx = 0)
            loss_g, log_g = self.loss(
                inputs, recons, # such that we can reuse the same loss function (latent rec loss has weight 0)
                inputs, recons,
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
                inputs.detach(), recons.detach(),
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
        Validation step for the VQ-VAE-2.
        Args:
            batch (torch.Tensor): Input batch tensor.
            batch_idx (int): Index of the current batch.
        Returns:
            torch.Tensor: Total loss for the current batch.
        """
        # -- 1. Prepare data -------------------------------------- #
        inputs = self.get_input(batch)
        recons, qloss, ind = self(inputs, return_pred_indices=True)

        # Reconstructions needed for FID / spectral score computation
        if self.track_fid or self.track_spectral:
            self._val_recons_img.append(recons.cpu())

        # -- 2. Compute losses ------------------------------------ #
        if isinstance(self.loss, VQLPIPSWithDiscriminator):
            loss_g, log_g = self.loss(
                inputs, recons,
                inputs, recons,
                qloss,
                optimizer_idx=0,
                global_step=self.global_step,
                predicted_indices=ind,
                split="val"
            )
            loss_d, log_d = self.loss(
                inputs.detach(), recons.detach(),
                inputs.detach(), recons.detach(),
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
        if isinstance(self.loss, VQLPIPSWithDiscriminator):
            # two-optimizer GAN setting
            d_params = list(self.loss.discriminator.parameters())
            d_ids    = set(map(id, d_params))
            g_params = [p for p in self.parameters() if id(p) not in d_ids]

            opt_g = torch.optim.Adam(g_params, lr=self.hparams.learning_rate, betas=(0.5,0.999))
            opt_d = torch.optim.Adam(d_params, lr=4*self.hparams.learning_rate, betas=(0.0,0.999))
            return [opt_g, opt_d]
        else:
            return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        
    def get_input(self, batch):
        """Get the input tensor from the batch and ensure its correct dtype."""
        return batch.to(memory_format=torch.contiguous_format).float()
    
    def on_train_start(self):
        """Grab a fixed mini-batch of images for logging."""
        if self.global_rank == 0:
            # Grab the first validation batch and keep up to 16 examples
            val_batch = next(iter(self.trainer.datamodule.val_dataloader()))
            images = val_batch[0] if isinstance(val_batch, (list, tuple)) else val_batch
            # Preserve batch dimension (B,C,H,W) and move to the right device
            self.fixed_images = images[:16].to(self.device)

    @torch.no_grad()
    def _log_reconstructions(self):
        """Log reconstructed images for visualization."""
        # Ensure tensor is on the correct device and has a batch dimension
        inputs = self.fixed_images.to(self.device)
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(0)
        recons = self(inputs, return_only_recon=True)

        # Create grid of images
        grid = make_grid(torch.cat([inputs, recons], dim=0), nrow=8, normalize=True)

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