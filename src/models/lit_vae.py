import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.distributed as dist
from torchvision.utils import make_grid

from src.models.modules.losses import SDVAELoss


class LitVAE(pl.LightningModule):
    """
    Lightning module for training a Variational Autoencoder (VAE).
    This module handles the forward pass, training step, validation step,
    and optimizer configuration for the VAE model.
    """
    def __init__(
            self,
            model,
            learning_rate=4.5e-6,
            fid_instance=None,
            spectral_instance=None,
            **loss_kwargs
        ):
        """
        Initializes the LitVAE module.
        Args:
            model: The VAE model to be trained.
            learning_rate (float): Learning rate for the optimizer.
            **loss_kwargs: Additional keyword arguments for loss configuration.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model" , "fid_instance", "spectral_instance"])
        self.learning_rate = learning_rate

        # Initialize loss
        self.loss = SDVAELoss(**loss_kwargs)
        self.loss.eval()

        # FID metric
        self.fid_instance = fid_instance
        self.track_fid = bool(self.fid_instance is not None)

        # Spectral metric
        self.spectral_instance = spectral_instance
        self.track_spectral = bool(self.spectral_instance is not None)

        # store model
        self.model = model

    def forward(self, x):
        # 1. Encode
        latents = self.model.encode(x).latent_dist
        # 2. Sample
        z = latents.sample()
        # 3. Decode
        recon = self.model.decode(z).sample
        return recon, latents
    
    def training_step(self, batch, batch_idx):
        # Forward pass
        inputs = batch
        recon, latents = self.forward(inputs)

        # Calculate the loss
        loss, log_dict = self.loss(
            inputs=inputs,
            recons=recon,
            latents=latents,
            split="train",
        )

        # Log losses
        self.log_dict(log_dict, on_step=True, on_epoch=False, sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        # Forward pass
        inputs = batch
        recon, latents = self.forward(inputs)

        # Store reconstructed images for metric calculation
        if self.track_fid or self.track_spectral:
            self._val_recons_img.append(recon.cpu())

        # Calculate the loss
        loss, log_dict = self.loss(
            inputs=inputs,
            recons=recon,
            latents=latents,
            split="val",
        )

        # Log losses
        self.log_dict(log_dict, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
    
    def on_before_optimizer_step(self, optimizer):
        # ensure DDP buckets see contiguous grads
        for p in self.parameters():
            if p.grad is not None and not p.grad.is_contiguous():
                p.grad = p.grad.contiguous()

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
        recons, _ = self(inputs)

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