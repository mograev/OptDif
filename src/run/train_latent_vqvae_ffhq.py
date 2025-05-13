import yaml
import argparse
import logging

import torch
import pytorch_lightning as pl
from torchvision import transforms

from src.dataloader.ffhq import FFHQWeightedDataset
from src.dataloader.weighting import DataWeighter
from src.models.latent_models import LatentVQVAE
from src.metrics.fid import FIDScore
from src.metrics.spectral import SpectralScore
from src.dataloader.utils import MultiModeDataset

from diffusers import AutoencoderKL

# Set the multiprocessing start method to spawn
import torch.multiprocessing as mp

# Set the float32 matmul precision to medium
# This is important for compatibility with certain hardware and software configurations
torch.set_float32_matmul_precision("medium")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    # -- Parse arguments ------------------------------------------ #

    # Parse arguments for external configuration
    parser = argparse.ArgumentParser()
    parser = FFHQWeightedDataset.add_data_args(parser)
    parser = DataWeighter.add_weight_args(parser)
    args = parser.parse_args()

    # Direct arguments
    args.seed=42
    args.max_epochs=100
    args.model_output_dir="models/latent_vqvae"
    args.model_config_path="models/latent_vqvae/configs/sd35m_to_512d_lpips_disc.yaml"

    # Seed everything
    pl.seed_everything(args.seed)

    # -- Load Data module ----------------------------------------- #

    # Load SD-VAE model
    sd_vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-3.5-medium", subfolder="vae")
    sd_vae.eval()

    # Load data
    datamodule = FFHQWeightedDataset(args, DataWeighter(args), sd_vae)
    datamodule.set_mode("img")
    
    # -- Initialize FID and Spectral metric ----------------------- #

    # Initialize instances
    fid_instance = FIDScore(img_size=256, device="cuda")
    spectral_instance = SpectralScore(img_size=256, device="cuda")

    # Load validation dataset
    val_filename_list = datamodule.val_dataloader().dataset.filename_list
    val_dataset = MultiModeDataset(
        val_filename_list,
        mode_dirs={"img": args.img_dir},
        transform=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]),
    )
    val_dataset.set_mode("img")

    # Fit metrics
    fid_instance.fit_real(val_dataset)
    spectral_instance.fit_real(val_dataset)

    # -- Load Latent VQVAE model ------------------------------- #

    # Load latent VQVAE config
    with open(args.model_config_path, "r") as f:
        model_config = yaml.safe_load(f)

    # Initialize LatentVQVAE model
    latent_vqvae = LatentVQVAE(
        ddconfig=model_config["ddconfig"],
        lossconfig=model_config["lossconfig"],
        learning_rate=model_config["learning_rate"],
        sd_vae_path="stabilityai/stable-diffusion-3.5-medium",
        ckpt_path=None,
        fid_instance=fid_instance,
        spectral_instance=spectral_instance,
    )

    # -- Initialize and run trainer ------------------------------- #

    # Progress bar
    pl._logger.setLevel(logging.INFO)

    # TensorBoard logger to log training progress
    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir=args.model_output_dir,
        version="version_7",
        name="",
    )

    # Model checkpoint callback to save model checkpoints
    checkpointer = pl.callbacks.ModelCheckpoint(
        filename="epoch_{epoch:03d}",
        auto_insert_metric_name=False,
        save_top_k=-1,
        every_n_epochs=1,
        save_last=True,
    )

    # Enable PyTorch anomaly detection
    with torch.autograd.set_detect_anomaly(True):
        # Create trainer
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=4,
            strategy="ddp_find_unused_parameters_true",  # required
            max_epochs=args.max_epochs,
            limit_train_batches=1.0,
            limit_val_batches=0.5,
            logger=tb_logger,
            callbacks=[checkpointer],
            enable_progress_bar=True,
        )

        # Fit model
        trainer.fit(latent_vqvae, datamodule)