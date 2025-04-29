from contextlib import contextmanager

import torch
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl

# Import Stable Diffusion VAE decoder for optional pixel-space losses
from diffusers import AutoencoderKL

from src.models.modules.autoencoder import Encoder, Decoder
from src.models.modules.distributions import DiagonalGaussianDistribution
from src.models.modules.quantize import VectorQuantizer2 as VectorQuantizer
from src.models.modules.utils import instantiate_from_config
from src.models.modules.losses import SimpleVAELoss, VAEWithDiscriminator, LPIPSWithDiscriminator
from src.models.modules.ema import LitEma


class LatentVAE(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 sd_vae_path=None,
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
        
        # Create encoder/decoder for variational autoencoder
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)

        # Create loss function
        self.loss = instantiate_from_config(lossconfig)

        # Manual optimization to allow multiple optimizers
        self.automatic_optimization = False
        
        # Ensure it is a VAE
        assert ddconfig["double_z"]

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

        if sd_vae_path:
            # Load weights from a pre-trained Stable Diffusion VAE
            self.sd_vae = AutoencoderKL.from_pretrained(sd_vae_path, subfolder="vae")
            self.sd_vae.eval()
            self.sd_vae.requires_grad_(False)

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
    
    @torch.no_grad()
    def latents_to_images(self, latents):
        """
        Decode Stable-Diffusion latents to pixel space.
        """
        if self.sd_vae is None:
            raise RuntimeError("latents_to_images() called but sd_vae is not initialized.")
        # Move decoder to correct device if necessary
        self.sd_vae.to(latents.device)
        images = self.sd_vae.decode(latents).sample
        return images
    
    def training_step(self, batch, batch_idx):
        # -- 1. Get data ----------------------------------------------------
        # Get inputs & reconstructions
        inputs = self.get_input(batch)
        recons, posterior = self(inputs)

        # Get the last layer of the decoder for adaptive loss
        last_layer = self.get_last_layer()

        # -- 2. Run training step based on the loss function ----------------

        if isinstance(self.loss, LPIPSWithDiscriminator):
            # -- A1. Decode latents to images -------------------------------
            try:
                with torch.no_grad():
                    inputs_img = self.latents_to_images(inputs)
                    recons_img = self.latents_to_images(recons)
            except AttributeError:
                raise RuntimeError("latents_to_images() called but sd_vae is not initialized.")

            # -- A2. Grab the two optimizers --------------------------------
            opt_vae, opt_disc = self.optimizers()

            # -- A3. Generator / VAE update (optimizer_idx = 0) -------------
            loss_gen, log_gen = self.loss(
                inputs, recons,
                inputs_img, recons_img,
                posterior,
                optimizer_idx=0,
                global_step=self.global_step,
                last_layer=last_layer,
                split="train"
            )

            opt_vae.zero_grad()
            self.manual_backward(loss_gen)
            opt_vae.step()

            # -- A4. Discriminator update (optimizer_idx = 1) ---------------
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

            # -- A5. Summarize losses ---------------------------------------
            log_dict = {**log_gen, **log_disc}
            total_loss = loss_gen + loss_disc

        elif isinstance(self.loss, VAEWithDiscriminator):
            # -- B1. Grab the two optimizers --------------------------------
            opt_vae, opt_disc = self.optimizers()

            # -- B2. Generator / VAE update (optimizer_idx = 0) -------------
            loss_gen, log_dict_gen = self.loss(
                inputs=inputs,
                reconstructions=recons,
                posterior=posterior,
                optimizer_idx=0,
                global_step=self.global_step,
                split="train"
            )

            opt_vae.zero_grad()
            self.manual_backward(loss_gen)  
            opt_vae.step()

            # -- B3. Discriminator update (optimizer_idx = 1) ---------------
            loss_disc, log_dict_disc = self.loss(
                inputs=inputs.detach(),
                reconstructions=recons.detach(),
                posterior=posterior.detach(),
                optimizer_idx=1,
                global_step=self.global_step,
                split="train"
            )
            opt_disc.zero_grad()
            self.manual_backward(loss_disc)
            opt_disc.step()

            # -- B4. Summarize losses ---------------------------------------
            log_dict = {**log_dict_gen, **log_dict_disc}
            total_loss = loss_gen + loss_disc
            
        elif isinstance(self.loss, SimpleVAELoss):
            # -- C1. Grab the optimizer -------------------------------------
            opt_vae = self.optimizers()

            # -- C2. Compute loss -------------------------------------------
            total_loss, log_dict = self.loss(
                inputs=inputs,
                reconstructions=recons,
                posterior=posterior,
                split="train"
            )

            opt_vae.zero_grad()
            self.manual_backward(total_loss)
            opt_vae.step()
        
        else:
            raise ValueError("Invalid loss function: {}".format(type(self.loss)))
        

        # -- 3. Logging -----------------------------------------------------
        # Log metrics
        self.log_dict(log_dict, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/total_loss", total_loss, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)

        return total_loss
    
    
    def validation_step(self, batch, batch_idx):
        # -- 1. Get data ----------------------------------------------------
        # Get inputs & reconstructions
        inputs = self.get_input(batch)
        recons, posterior = self(inputs)

        # Get the last layer of the decoder for adaptive loss
        last_layer = self.get_last_layer()

        # -- 2. Run training step based on the loss function ----------------

        if isinstance(self.loss, LPIPSWithDiscriminator):
            # -- A1. Decode latents to images -------------------------------
            try:
                with torch.no_grad():
                    inputs_img = self.latents_to_images(inputs)
                    recons_img = self.latents_to_images(recons)
            except AttributeError:
                raise RuntimeError("latents_to_images() called but sd_vae is not initialized.")

            # -- A2. Grab the two optimizers --------------------------------
            opt_vae, opt_disc = self.optimizers()

            # -- A3. Generator / VAE loss (optimizer_idx = 0) ---------------
            loss_gen, log_gen = self.loss(
                inputs, recons,
                inputs_img, recons_img,
                posterior,
                optimizer_idx=0,
                global_step=self.global_step,
                last_layer=last_layer,
                split="val"
            )

            # -- A4. Discriminator loss (optimizer_idx = 1) -----------------
            loss_disc, log_disc = self.loss(
                inputs.detach(), recons.detach(),
                inputs_img.detach(), recons_img.detach(),
                posterior.detach(),
                optimizer_idx=1,
                global_step=self.global_step,
                split="val"
            )

            # -- A5. Summarize losses ---------------------------------------
            log_dict = {**log_gen, **log_disc}
            total_loss = loss_gen + loss_disc

        elif isinstance(self.loss, VAEWithDiscriminator):
            # -- B1. Grab the two optimizers --------------------------------
            opt_vae, opt_disc = self.optimizers()

            # -- B2. Generator / VAE update (optimizer_idx = 0) -------------
            loss_gen, log_dict_gen = self.loss(
                inputs=inputs,
                reconstructions=recons,
                posterior=posterior,
                optimizer_idx=0,
                global_step=self.global_step,
                split="val"
            )

            # -- B3. Discriminator update (optimizer_idx = 1) ---------------
            loss_disc, log_dict_disc = self.loss(
                inputs=inputs.detach(),
                reconstructions=recons.detach(),
                posterior=posterior.detach(),
                optimizer_idx=1,
                global_step=self.global_step,
                split="val"
            )

            # -- B4. Summarize losses ---------------------------------------
            log_dict = {**log_dict_gen, **log_dict_disc}
            total_loss = loss_gen + loss_disc
            
        elif isinstance(self.loss, SimpleVAELoss):
            # -- C1. Grab the optimizer -------------------------------------
            opt_vae = self.optimizers()

            # -- C2. Compute loss -------------------------------------------
            total_loss, log_dict = self.loss(
                inputs=inputs,
                reconstructions=recons,
                posterior=posterior,
                split="val"
            )
        
        else:
            raise ValueError("Invalid loss function: {}".format(type(self.loss)))


        # -- 5. Logging -----------------------------------------------------
        # Summarize losses
        log_dict = {**log_gen, **log_disc}
        total_loss = loss_gen + loss_disc

        # Log metrics
        self.log_dict(log_dict, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/total_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        return total_loss


    def configure_optimizers(self):

        if isinstance(self.loss, (VAEWithDiscriminator, LPIPSWithDiscriminator)):
            # Get relevant parameters for VAE and discriminator
            disc_params = list(self.loss.discriminator.parameters())
            disc_ids = set(id(p) for p in disc_params)
            vae_params = [p for p in self.parameters() if id(p) not in disc_ids]

            # Use two optimizers for VAE and discriminator
            # First optimizer (generator / VAE)
            opt_vae = torch.optim.Adam(
                vae_params,
                lr=self.learning_rate, 
                betas=(0.5, 0.999)
            )

            # Second optimizer (discriminator)
            opt_disc = torch.optim.Adam(
                disc_params,
                lr=self.learning_rate, 
                betas=(0.5, 0.999)
            )

            # Return both optimizers
            return [opt_vae, opt_disc]

        elif isinstance(self.loss, SimpleVAELoss):
            # Use a single optimizer for VAE
            return torch.optim.Adam(
                self.parameters(), 
                lr=self.learning_rate
            )
        
        else:
            raise ValueError("Invalid loss function: {}".format(type(self.loss)))
    
    def get_last_layer(self):
        return self.decoder.conv_out if hasattr(self.decoder, 'conv_out') else None
    


class LatentVQVAE(pl.LightningModule):
    def __init__(self, ddconfig, lossconfig, n_embed, embed_dim, ckpt_path=None, ignore_keys=[], monitor=None, scheduler_config=None, remap=None, sane_index_shape=False, use_ema=False):

        super().__init__()
        self.n_embed = n_embed
        self.embed_dim = embed_dim

        # Create encoder/decoder for VQVAE
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)

        # Create loss function
        self.loss = instantiate_from_config(lossconfig)
        self.save_hyperparameters()
        self.learning_rate = 1e-4
        self.embed_dim = embed_dim

        self.automatic_optimization = True

        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.05,
                                        remap=remap,
                                        sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        if monitor is not None:
            self.monitor = monitor

        self.scheduler_config = scheduler_config

        if use_ema:
            self.model_ema = LitEma(self)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")
        self.use_ema = use_ema

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize(code_b)[0]
        dec = self.decode(quant_b)
        return dec
    
    def forward(self, input, return_pred_indices=False):
        quant, diff, (_,_,ind) = self.encode(input)
        dec = self.decode(quant)
        if return_pred_indices:
            return dec, diff, ind
        return dec, diff

    def get_input(self, batch):
        return batch.to(memory_format=torch.contiguous_format).float()

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch)
        xrec, qloss = self(x)

        total_loss, log_dict = self.loss(x, xrec, qloss)

        # Log metrics
        for k, v in log_dict.items():
            self.log(f"train_{k}", v, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True) #, sync_dist=True
        self.log("train_total_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True) #, sync_dist=True
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch)
        xrec, qloss = self(x)

        total_loss, log_dict = self.loss(x, xrec, qloss)

        # Log metrics
        for k, v in log_dict.items():
            self.log(f"val_{k}", v, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True) #, sync_dist=True
        self.log("val_total_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True) #, sync_dist=True

        return total_loss

    def configure_optimizers(self):
        # Create optimizer
        opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.999))

        # Create scheduler if specified
        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)
            scheduler = LambdaLR(opt, lr_lambda=scheduler.get_lr_lambda)
            return [opt], [scheduler]

        return opt

    def get_last_layer(self):
        return self.decoder.conv_out.weight


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