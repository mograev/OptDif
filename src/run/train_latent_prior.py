import argparse
import logging
import yaml
import torch
import torch.multiprocessing as mp
import pytorch_lightning as pl
from torchvision import transforms

from src.dataloader.ffhq import FFHQDataset
from src.dataloader.utils import OptEncodeDataset
from src.metrics.fid import FIDScore
from src.models.transformer_prior import HierarchicalTransformerPrior
from src.models.pixelsnail_prior import HierarchicalPixelSnailPrior
from src.models.latent_models import LatentVQVAE2
from src.models.vqvae2 import VQVAE2

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
    parser = FFHQDataset.add_data_args(parser)

    # Shared arguments
    parser.add_argument("--latent_model_config_path", required=True, help="YAML config used to build the LatentVQVAE2")
    parser.add_argument("--latent_model_ckpt_path", required=True, help="Path to *.ckpt of your trained LatentVQVAE2")
    parser.add_argument("--prior_type", type=str, choices=["transformer", "pixelsnail"], default="transformer", help="Type of prior model to use")
    parser.add_argument("--prior_out_dir", required=True, help="Directory to save the prior model")
    parser.add_argument("--prior_version", help="Version under which to save the prior model")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads in the Transformer prior")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_devices", type=int, default=4)
    # Transformer hyper-params
    parser.add_argument("--d_model", type=int, default=512, help="Dimension of the model in the Transformer prior")
    parser.add_argument("--n_layers", type=int, default=8, help="Number of layers in the Transformer prior")
    # PixelSnail hyper-params
    parser.add_argument("--n_chan", type=int, default=256, help="Number of channels in the PixelSnail prior")
    parser.add_argument("--n_blocks", type=int, default=8, help="Number of blocks in the PixelSnail prior")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate in the PixelSnail prior")
    args = parser.parse_args()

    # Seed everything
    pl.seed_everything(42)

    # -- Load Data module ----------------------------------------- #

    # Load SD-VAE model
    sd_vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-3.5-medium", subfolder="vae")
    sd_vae.eval()

    # Load data
    datamodule = FFHQDataset(
        args,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop(size=256, scale=(0.9, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]),
    ).set_encode(False)

    # -- Initialize FID metric ------------------------------------ #

    # Initialize FID instance
    fid_instance = FIDScore(img_size=256, device=args.device)

    # Load validation dataset
    val_filename_list = datamodule.val_dataloader().dataset.filename_list
    val_dataset = OptEncodeDataset(
        val_filename_list,
        img_dir=datamodule.img_dir,
        transform=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]),
    )

    # Fit metric
    fid_instance.fit_real(val_dataset)

    # -- Load latent model ---------------------------------------- #
    with open(args.latent_model_config_path) as f:
        config = yaml.safe_load(f)

    latent_model = LatentVQVAE2(
        ddconfig=config["ddconfig"],
        lossconfig=config["lossconfig"],
        learning_rate=0.0,
        sd_vae_path="stabilityai/stable-diffusion-3.5-medium",
        fid_instance=fid_instance,
        spectral_instance=None,
    )
    # latent_model = VQVAE2(
    #     ddconfig=config["ddconfig"],
    #     lossconfig=config["lossconfig"],
    #     learning_rate=0.0,
    #     fid_instance=fid_instance,
    #     spectral_instance=None,
    # )

    state = torch.load(args.latent_model_ckpt_path, map_location="cpu")["state_dict"]
    latent_model.load_state_dict(state, strict=True)
    latent_model.eval()
    latent_model.requires_grad_(False)

    # -- Initialize prior ----------------------------------------- #

    if args.prior_type == "transformer":
        prior = HierarchicalTransformerPrior(
            vqvae=latent_model,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    elif args.prior_type == "pixelsnail":
        prior = HierarchicalPixelSnailPrior(
            vqvae=latent_model,
            n_chan=args.n_chan,
            n_blocks=args.n_blocks,
            lr=args.lr,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
        )

    # -- Initialize and run trainer ------------------------------- #

    # Progress bar and logging
    pl._logger.setLevel(logging.INFO)

    # Tensorboard logger to log training progress
    logger = pl.loggers.TensorBoardLogger(
        save_dir=args.prior_out_dir,
        version=f"version_{args.prior_version}",
        name="",
    )

    # Model checkpoint callback to save model checkpoints
    ckpt = pl.callbacks.ModelCheckpoint(
        filename="epoch_{epoch:03d}",
        auto_insert_metric_name=False,
        save_last=True,
        save_top_k=-1,
        every_n_epochs=1,
    )

    # Enable pytorch anomaly detection
    with torch.autograd.set_detect_anomaly(True):
        # Create trainer
        trainer = pl.Trainer(
            accelerator="gpu" if args.device == "cuda" else "cpu",
            devices=args.num_devices if args.device == "cuda" else 1,
            strategy="ddp_find_unused_parameters_true" if args.num_devices > 1 else None,
            max_epochs=args.max_epochs,
            logger=logger,
            callbacks=[ckpt],
            limit_train_batches=1.0,
            limit_val_batches=0.5,
            gradient_clip_val=1.0,
            accumulate_grad_batches=1,
            enable_progress_bar=False,
        )

        # Fit model
        trainer.fit(prior, datamodule)