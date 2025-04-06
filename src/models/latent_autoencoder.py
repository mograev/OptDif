import torch
import pytorch_lightning as pl

from src.models.modules.autoencoder import Encoder, Decoder
from src.models.modules.utils import instantiate_from_config

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
        latent_ddconfig = ddconfig.copy()

        # Create encoder/decoder for non-variational autoencoder
        self.encoder = Encoder(**latent_ddconfig)
        self.decoder = Decoder(**latent_ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.automatic_optimization = True  # single optimizer now

        # Ensure it is an Autoencoder (not a VAE)
        assert not latent_ddconfig["double_z"]

        # Calculate spatial dimensions
        bottleneck_resolution = latent_ddconfig["resolution"] // (
            2 ** (len(latent_ddconfig["ch_mult"]) - 1)
        )
        spatial_size = bottleneck_resolution * bottleneck_resolution

        # Single set of features for 1D latent
        self.quant_conv = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(
                latent_ddconfig["z_channels"] * spatial_size,
                embed_dim
            )
        )

        # De-project from 1D latent
        self.post_quant_conv = torch.nn.Sequential(
            torch.nn.Linear(
                embed_dim,
                latent_ddconfig["z_channels"] * spatial_size
            ),
            torch.nn.Unflatten(
                1,
                (latent_ddconfig["z_channels"], bottleneck_resolution, bottleneck_resolution)
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
        loss, log_dict = self.loss(inputs=inputs, reconstructions=reconstructions)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch)
        reconstructions = self(inputs)
        val_loss, log_dict = self.loss(inputs=inputs, reconstructions=reconstructions)
        self.log("val_loss", val_loss, prog_bar=True, on_step=False, on_epoch=True)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def log_images(self, batch, only_inputs=False, **kwargs):
        log = {}
        latents = self.get_input(batch).to(self.device)
        if not only_inputs:
            recon = self(latents)
            log["reconstructions"] = recon
            log["rec_error"] = torch.abs(latents - recon)
        log["inputs"] = latents
        return log
