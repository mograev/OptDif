import yaml
import argparse
import logging

import torch
import pytorch_lightning as pl

from src.dataloader.ffhq import FFHQWeightedDataset
from src.dataloader.weighting import DataWeighter
from src.models.latent_models import LatentVQVAE

from diffusers import AutoencoderKL

# Set the multiprocessing start method to spawn
import torch.multiprocessing as mp

# Set the float32 matmul precision to medium
# This is important for compatibility with certain hardware and software configurations
torch.set_float32_matmul_precision("medium")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    # Parse arguments for external configuration
    parser = argparse.ArgumentParser()
    parser = FFHQWeightedDataset.add_data_args(parser)
    parser = DataWeighter.add_weight_args(parser)
    args = parser.parse_args()

    # Direct arguments
    args.seed=42
    args.device="cuda"
    args.data_device="cuda"
    args.max_epochs=100
    args.model_output_dir="/pfs/work9/workspace/scratch/ma_mgraevin-optdif/models/latent_vqvae"
    args.model_config_path="/pfs/work9/workspace/scratch/ma_mgraevin-optdif/models/latent_vqvae/configs/sd35m_to_512d.yaml"
    # args.chkpt_path="/pfs/work9/workspace/scratch/ma_mgraevin-optdif/models/latent_vqvae/version_1/checkpoints/last.ckpt"

    # Seed everything
    pl.seed_everything(args.seed)

    # Load SD-VAE model
    sd_vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-3.5-medium", subfolder="vae")
    sd_vae.eval()

    # Load data
    datamodule = FFHQWeightedDataset(args, DataWeighter(args), sd_vae)
    datamodule.set_mode("img_tensor")

    # Load latent VAE config
    with open(args.model_config_path, "r") as f:
        model_config = yaml.safe_load(f)

    # Initialize LatentVQVAE model
    latent_vqvae = LatentVQVAE(
        ddconfig=model_config["ddconfig"],
        lossconfig=model_config["lossconfig"],
        n_embed=model_config["embedconfig"]["n_embed"],
        embed_dim=model_config["embedconfig"]["embed_dim"],
        ckpt_path=None,
        monitor="val_total_loss",
    )

    # Progress bar
    pl._logger.setLevel(logging.INFO)

    # TensorBoard logger to log training progress
    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir=args.model_output_dir,
        version="version_3",
        name="",
    )

    # Model checkpoint callback to save the best model
    checkpointer = pl.callbacks.ModelCheckpoint(save_last=True, monitor="val_total_loss")

    # Enable PyTorch anomaly detection
    with torch.autograd.set_detect_anomaly(True):
        # Create trainer
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=4,
            strategy="ddp",
            max_epochs=args.max_epochs,
            limit_train_batches=1.0,
            limit_val_batches=1.0,
            logger=tb_logger,
            callbacks=[checkpointer],
            enable_progress_bar=False,
        )

        # Fit model
        trainer.fit(latent_vqvae, datamodule)