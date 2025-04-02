import json
import argparse
import logging

import torch
import pytorch_lightning as pl

from src.dataloader.ffhq import FFHQWeightedDataset
from src.dataloader.weighting import DataWeighter
from src.models.latent_vae import LatentVAE


# Parse arguments for external configuration
parser = argparse.ArgumentParser()
parser = FFHQWeightedDataset.add_data_args(parser)
parser = DataWeighter.add_weight_args(parser)
args = parser.parse_args()

# Direct arguments
args.sd_latent_dir="/pfs/work7/workspace/scratch/ma_mgraevin-optdif/data/ffhq/sd_latents"
args.device="cuda"
args.seed=42
args.max_epochs=100
args.model_output_dir="/pfs/work7/workspace/scratch/ma_mgraevin-optdif/models/latent_vae"
args.latent_vae_config_path="/pfs/work7/workspace/scratch/ma_mgraevin-optdif/models/latent_vae/configs/sd35m_to_128d.json"

# Seed everything
pl.seed_everything(args.seed)

# Load data
datamodule = FFHQWeightedDataset(args, DataWeighter(args))
datamodule.add_mode("sd_latent", args.sd_latent_dir, set_mode=True)

# Load latent VAE config
with open(args.latent_vae_config_path, "r") as f:
    latent_vae_config = json.load(f)

# Initialize LatentVAE model
latent_vae = LatentVAE(
    ddconfig=latent_vae_config["ddconfig"],
    lossconfig=latent_vae_config["lossconfig"],
    embed_dim=latent_vae_config["embed_dim"],
    ckpt_path=None,
    monitor="val_total_loss",
)

# Progress bar
pl._logger.setLevel(logging.CRITICAL)
train_pbar = pl.callbacks.TQDMProgressBar(
    refresh_rate=1,
    leave=True,
)

# TensorBoard logger to log training progress
tb_logger = pl.loggers.TensorBoardLogger(save_dir=args.model_output_dir)

# Model checkpoint callback to save the best model
checkpointer = pl.callbacks.ModelCheckpoint(save_last=True, monitor="val_total_loss")

# Enable PyTorch anomaly detection
with torch.autograd.set_detect_anomaly(True):
    # Create trainer
    trainer = pl.Trainer(
        accelerator="gpu" if args.device == "cuda" else "cpu",
        max_epochs=args.max_epochs,
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        logger=tb_logger,
        callbacks=[train_pbar, checkpointer],
    )

    # Fit model
    trainer.fit(latent_vae, datamodule)