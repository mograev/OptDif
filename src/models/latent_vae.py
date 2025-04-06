import torch
import pytorch_lightning as pl

from src.models.modules.autoencoder import Encoder, Decoder
from src.models.modules.distributions import DiagonalGaussianDistribution
from src.models.modules.utils import instantiate_from_config
from src.models.modules.losses import SimpleVAELoss, VAEWithDiscriminator


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

        # Manual optimization to allow multiple optimizers
        self.automatic_optimization = False
        
        # Ensure it is a VAE
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
        # Get inputs & reconstructions
        inputs = self.get_input(batch)
        reconstructions, posterior = self(inputs)

        if isinstance(self.loss, VAEWithDiscriminator):
            # A1) Get optimizers
            opt_vae, opt_disc = self.optimizers()

            # A2) Generator forward + backward
            loss_gen, log_dict_gen = self.loss(
                inputs=inputs,
                reconstructions=reconstructions,
                posterior=posterior,
                optimizer_idx=0,
                global_step=self.global_step
            )
            opt_vae.zero_grad()
            self.manual_backward(loss_gen)  
            opt_vae.step()

            # A3) Discriminator forward + backward (detach reconstructions)
            with torch.no_grad():
                reconstructions, posterior = self(inputs) # recompute reconstructions
                reconstructions = reconstructions.detach()
                # Rebuild distribution with detached mu and logvar
                posterior = DiagonalGaussianDistribution(
                    posterior.parameters.detach()
                )
            loss_disc, log_dict_disc = self.loss(
                inputs=inputs,
                reconstructions=reconstructions,
                posterior=posterior,
                optimizer_idx=1,
                global_step=self.global_step
            )
            opt_disc.zero_grad()
            self.manual_backward(loss_disc)
            opt_disc.step()

            # A4) Summarize losses
            log_dict = {**log_dict_gen, **log_dict_disc}
            total_loss = loss_gen + loss_disc
            
        elif isinstance(self.loss, SimpleVAELoss):
            # B1) Get optimizer
            opt_vae = self.optimizers()

            # B2) Compute loss
            total_loss, log_dict = self.loss(
                inputs=inputs,
                reconstructions=reconstructions,
                posterior=posterior
            )

            # B3) Backward pass
            opt_vae.zero_grad()
            self.manual_backward(total_loss)
            opt_vae.step()
        
        else:
            raise ValueError("Invalid loss function: {}".format(type(self.loss)))
                
        # Log metrics
        for k, v in log_dict.items():
            self.log(k, v, prog_bar=True, on_step=True, on_epoch=True)
        self.log("total_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True)

        return total_loss
    
    def validation_step(self, batch, batch_idx):
        # Get inputs & reconstructions
        inputs = self.get_input(batch)
        reconstructions, posterior = self(inputs)
        
        if isinstance(self.loss, VAEWithDiscriminator):
            # A) Generator loss
            loss_gen, log_dict_gen = self.loss(
                inputs=inputs,
                reconstructions=reconstructions,
                posterior=posterior,
                optimizer_idx=0,
                global_step=self.global_step
            )

            # A) Discriminator loss
            loss_disc, log_dict_disc = self.loss(
                inputs=inputs,
                reconstructions=reconstructions,
                posterior=posterior,
                optimizer_idx=1,
                global_step=self.global_step
            )

            # A) Summarize losses
            log_dict = {**log_dict_gen, **log_dict_disc}
            total_loss = loss_gen + loss_disc
        
        elif isinstance(self.loss, SimpleVAELoss):
            # B) Compute loss
            total_loss, log_dict = self.loss(
                inputs=inputs,
                reconstructions=reconstructions,
                posterior=posterior
            )

        else:
            raise ValueError("Invalid loss function: {}".format(type(self.loss)))

        # Log metrics
        for k, v in log_dict.items():
            self.log(f"val_{k}", v, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_total_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True)

        return total_loss

    def configure_optimizers(self):

        if isinstance(self.loss, VAEWithDiscriminator):
            # Use two optimizers for VAE and discriminator
            # First optimizer (generator / VAE)
            opt_vae = torch.optim.Adam(
                self.parameters(), 
                lr=self.learning_rate, 
                betas=(0.5, 0.999)
            )

            # Second optimizer (discriminator)
            opt_disc = torch.optim.Adam(
                self.parameters(), 
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