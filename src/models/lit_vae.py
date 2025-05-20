import torch
import torch.nn.functional as F
import pytorch_lightning as pl


class LitVAE(pl.LightningModule):

    def __init__(self, model, lr=1e-4, beta=1.0):
        super().__init__()
        self.save_hyperparameters(ignore="model")
        self.lr = lr
        self.beta = beta

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
        x = batch
        recon, latents = self.forward(x)

        # Reconstruction loss
        recon_loss = F.mse_loss(recon, x, reduction="mean")

        # KL divergence
        kl_loss = -0.5 * torch.sum(
            1 + latents.logvar - latents.mean.pow(2) - latents.var
        ) / x.shape[0]

        # Total loss
        loss = recon_loss + self.beta * kl_loss

        # Log losses
        self.log_dict({
            "train/total_loss": loss,
            "train/recon_loss": recon_loss,
            "train/kl_loss": kl_loss
        })

        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
    
    def validation_step(self, batch, batch_idx):
        # Forward pass
        x = batch
        recon, latents = self.forward(x)

        # Reconstruction loss
        recon_loss = F.mse_loss(recon, x, reduction="mean")

        # KL divergence
        kl_loss = -0.5 * torch.sum(
            1 + latents.logvar - latents.mean.pow(2) - latents.var
        ) / x.shape[0]

        # Total loss
        loss = recon_loss + self.beta * kl_loss

        # Log losses
        self.log_dict({
            "val/total_loss": loss,
            "val/recon_loss": recon_loss,
            "val/kl_loss": kl_loss
        }, prog_bar=True)

        return loss