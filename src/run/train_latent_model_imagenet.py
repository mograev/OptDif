"""
ImageNet trainer for latent SD models (AE, VAE, VQ-VAE, Linear AE).
Lightning DDP, FID/Spectral fit, TensorBoard & checkpoints.
"""

import os
import yaml
import argparse
import logging

import torch
import pytorch_lightning as pl
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset

from src.dataloader.imagenet import ImageNetDataset
from src.models.latent_models import LatentVAE, LatentVQVAE, LatentAutoencoder, LatentLinearAE
from src.metrics.fid import FIDScore
from src.metrics.spectral import SpectralScore

from diffusers import AutoencoderKL

# Set the multiprocessing start method to spawn
import torch.multiprocessing as mp

# Set the float32 matmul precision to medium
# This is important for compatibility with certain hardware and software configurations
torch.set_float32_matmul_precision("medium")

# Helper class to drop labels from the dataset
class ImageOnlyDataset(Dataset):
    """Drop the label and only yield the image tensor."""
    def __init__(self, ds):
        self.ds = ds
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        img, _ = self.ds[idx]
        return img

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    # -- Parse arguments ------------------------------------------ #

    # Parse arguments for external configuration
    parser = argparse.ArgumentParser()
    parser = ImageNetDataset.add_data_args(parser)

    # Direct arguments
    parser.add_argument("--model_type", type=str, choices=["LatentVAE", "LatentVQVAE", "LatentAutoencoder", "LatentLinearAE"], help="Type of latent model to use")
    parser.add_argument("--model_version", type=str, help="Version of the latent model to use")
    parser.add_argument("--model_config_path", type=str, help="Path to the model config file")
    parser.add_argument("--model_output_dir", type=str, help="Directory to save the model")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of epochs to train the model")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training (e.g., 'cuda' or 'cpu')")
    parser.add_argument("--num_devices", type=int, default=4, help="Number of devices to use for training")
    args = parser.parse_args()

    # Seed everything
    pl.seed_everything(42)

    # -- Load Data module ----------------------------------------- #

    # Load SD-VAE model
    sd_vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-3.5-medium", subfolder="vae")
    sd_vae.eval()

    # Load data
    datamodule = ImageNetDataset(args, sd_vae)

    # -- Initialize FID and Spectral metric ----------------------- #

    # Initialize instances
    fid_instance = FIDScore(img_size=256, device=args.device, batch_size=args.batch_size, num_workers=args.num_workers)
    spectral_instance = SpectralScore(img_size=256, device=args.device, batch_size=args.batch_size, num_workers=args.num_workers)

    # Load validation dataset (workaround, no encoding)
    val_dir = os.path.join(args.img_dir, "val")
    val_dataset_raw = ImageFolder(
        root=val_dir,
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
    )

    # Drop labels from the validation dataset
    val_dataset = ImageOnlyDataset(val_dataset_raw)

    # Fit metrics
    fid_instance.fit_real(val_dataset)
    spectral_instance.fit_real(val_dataset)

    # -- Load latent model ---------------------------------------- #

    # Load model config
    with open(args.model_config_path, "r") as f:
        model_config = yaml.safe_load(f)

    # Initialize the correct latent model based on the CLI flag
    model_cls = {
        "LatentVAE": LatentVAE,
        "LatentVQVAE": LatentVQVAE,
        "LatentAutoencoder": LatentAutoencoder,
        "LatentLinearAE": LatentLinearAE,
    }[args.model_type]

    # Instantiate with the loaded configuration
    model = model_cls(
        ddconfig=model_config["ddconfig"],
        lossconfig=model_config["lossconfig"],
        learning_rate=model_config["learning_rate"],
        sd_vae_path="stabilityai/stable-diffusion-3.5-medium",
        fid_instance=fid_instance,
        spectral_instance=spectral_instance,
    )

    # -- Initialize and run trainer ------------------------------- #

    # Progress bar
    pl._logger.setLevel(logging.INFO)

    # TensorBoard logger to log training progress
    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir=args.model_output_dir,
        version=f"version_{args.model_version}",
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
            devices=args.num_devices if args.device == "cuda" else 1,
            strategy="ddp_find_unused_parameters_true" if args.device == "cuda" else None,
            max_epochs=args.max_epochs,
            limit_train_batches=0.05, # 5% ≈ 65000 images
            limit_val_batches=0.1, # 10% ≈ 5000 images
            logger=tb_logger,
            callbacks=[checkpointer],
            enable_progress_bar=False,
        )

        # Fit model
        trainer.fit(model, datamodule)