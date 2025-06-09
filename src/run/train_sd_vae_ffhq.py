import yaml
import argparse
import logging

import torch
import pytorch_lightning as pl
from torchvision import transforms

from src.dataloader.ffhq import FFHQWeightedDataset
from src.dataloader.weighting import DataWeighter
from src.metrics.fid import FIDScore
from src.metrics.spectral import SpectralScore
from src.dataloader.utils import MultiModeDataset
from src.models.lit_vae import LitVAE

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

    # Direct arguments
    parser.add_argument("--model_path", type=str, help="Path to the latent model")
    parser.add_argument("--model_version", type=str, default="v1", help="Version of the model to use")
    parser.add_argument("--model_output_dir", type=str, help="Directory to save the model")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of epochs to train the model")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training (e.g., 'cuda' or 'cpu')")
    parser.add_argument("--num_devices", type=int, default=4, help="Number of devices to use for training")
    args = parser.parse_args()

    # Seed everything
    pl.seed_everything(42)

    # -- Load Data module ----------------------------------------- #

    # Load data
    datamodule = FFHQWeightedDataset(args, DataWeighter(args))
    datamodule.set_mode("img")

    # -- Initialize FID and Spectral metric ----------------------- #

    # Initialize instances
    fid_instance = FIDScore(img_size=256, device=args.device, batch_size=args.batch_size, num_workers=args.num_workers)
    spectral_instance = SpectralScore(img_size=256, device=args.device, batch_size=args.batch_size, num_workers=args.num_workers)

    # Load validation dataset (workaround)
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

    # -- Load model ----------------------------------------------- #

    # Load SD-VAE model
    sd_vae = AutoencoderKL.from_pretrained(args.model_path, subfolder="vae")
    sd_vae.train()
    sd_vae.requires_grad_(True)

    # Wrap in Lightning module
    model = LitVAE(
        model=sd_vae,
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
            accelerator="gpu" if args.device == "cuda" else "cpu",
            devices=args.num_devices if args.device == "cuda" else 1,
            strategy="ddp_find_unused_parameters_true" if args.device == "cuda" else None,
            max_epochs=args.max_epochs,
            limit_train_batches=1.0,
            limit_val_batches=0.5,
            logger=tb_logger,
            callbacks=[checkpointer],
            enable_progress_bar=False,
        )

        # Fit model
        trainer.fit(model, datamodule)

    # -- Save final model ----------------------------------------- #

    model.model.save_pretrained(args.model_output_dir + f"version_{args.model_version}/huggingface")