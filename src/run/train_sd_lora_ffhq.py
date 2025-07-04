import argparse
from omegaconf import OmegaConf
from functools import reduce
from pathlib import Path
from itertools import islice

import numpy as np
import torch
from torch import Generator
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.transforms.functional as TF
import torchvision.utils as vutils
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers.optimization import get_scheduler
import einops

from src.dataloader.ffhq import FFHQDataset
from src.metrics.fid import FIDScore
from src.metrics.spectral import SpectralScore
from src.ctrloralter.model import SD15
from src.ctrloralter.annotators.openclip import VisionModel
from src.ctrloralter.annotators.midas import DepthEstimator
from src.ctrloralter.annotators.hed import TorchHEDdetector
from src.ctrloralter.mapper_network import SimpleMapper, FixedStructureMapper15
from src.ctrloralter.utils import add_lora_from_config, save_checkpoint

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

    # Direct arguments
    parser.add_argument("--model_version", type=str, default="v1", help="Version of the model to use")
    parser.add_argument("--model_output_dir", type=str, help="Directory to save the model")
    parser.add_argument("--struct_adapter", type=str, default=None, choices=["depth", "hed", "none"], help="Type of structure adapter to use")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of epochs to train the model")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training (e.g., 'cuda' or 'cpu')")
    parser.add_argument("--num_devices", type=int, default=4, help="Number of devices to use for training")
    args = parser.parse_args()

    output_dir = Path(args.model_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Seed everything
    pl.seed_everything(42)

    # Setup logger
    logger = get_logger("ctrloralter_ffhq")
    log_every_n_steps = 50

    # -- Load Data module ----------------------------------------- #

    # Load data
    datamodule = FFHQDataset(
        args,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop(size=(512, 512), scale=(0.9, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]),
    )

    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()

    # -- Setup Accelerator ---------------------------------------- #

    accelerator = Accelerator(
        project_dir=output_dir,
        log_with="tensorboard",
        gradient_accumulation_steps=1,
        mixed_precision="bf16",
    )

    # Init trackers
    logger.info("init trackers")
    if accelerator.is_main_process:
        accelerator.init_trackers("")

    
    # -- Initialize FID and Spectral metric ----------------------- #

    if accelerator.is_main_process:

        # Initialize instances
        fid_instance = FIDScore(img_size=512, device=args.device)
        spectral_instance = SpectralScore(img_size=512, device=args.device)

        # Load validation dataset
        val_dataset = datamodule.val_dataloader().dataset

        # Fit metrics
        fid_instance.fit_real(val_dataset)
        spectral_instance.fit_real(val_dataset)

    else:
        fid_instance = spectral_instance = None

    # Wait for all processes to finish fitting the metrics
    accelerator.wait_for_everyone()

    # -- Load model ----------------------------------------------- #

    # Load SD base model
    model = SD15(
        pipeline_type="diffusers.StableDiffusionPipeline",
        model_name="runwayml/stable-diffusion-v1-5",
        local_files_only=False,
    ).to(accelerator.device)

    # Load style adapter
    raw_cfg = {
        "ckpt_path": "ctrloralter/checkpoints",
        "lora": {
            "style": {
                "enable": "always",
                "optimize": True,
                # "ckpt_path": "/BS/optdif/work/models/sd_lora/version_2/checkpoints/epoch_036", # start with already fine-tuned style LoRA
                "ckpt_path": "ctrloralter/checkpoints/sd15-style-cross-160-h",
                "ignore_check": False,
                "cfg": True,
                "transforms": [],
                "config": {
                    "lora_scale": 1.0,
                    "rank": 160,
                    "c_dim": 1024,
                    "adaption_mode": "only_cross",
                    "lora_cls": "SimpleLoraLinear",
                    "broadcast_tokens": True,
                },
                "encoder": VisionModel(clip_model="laion/CLIP-ViT-H-14-laion2B-s32B-b79K", local_files_only=False),
                "mapper_network": SimpleMapper(1024, 1024),
            },
        }
    }
    # Optionally load structure adapter
    if args.struct_adapter == "depth":
        raw_cfg["lora"]["struct"] = {
                "enable": "always",
                "optimize": True,
                "ckpt_path": "ctrloralter/checkpoints/sd15-depth-128-only-res",
                "ignore_check": False,
                "cfg": False,
                "transforms": [],
                "config": {
                    "lora_scale": 1.0,
                    "rank": 128,
                    "c_dim": 128,
                    "adaption_mode": "only_res_conv",
                    "lora_cls": "NewStructLoRAConv",
                },
                "encoder": DepthEstimator(size=512, local_files_only=False),
                "mapper_network": FixedStructureMapper15(c_dim=128),
            }
    elif args.struct_adapter == "hed":
        raw_cfg["lora"]["struct"] = {
                "enable": "always",
                "optimize": True,
                "ckpt_path": "ctrloralter/checkpoints/sd15-hed-128-only-res",
                "ignore_check": False,
                "cfg": False,
                "transforms": [],
                "config": {
                    "lora_scale": 1.0,
                    "rank": 128,
                    "c_dim": 128,
                    "adaption_mode": "only_res_conv",
                    "lora_cls": "NewStructLoRAConv",
                },
                "encoder": TorchHEDdetector(size=512, local_files_only=False),
                "mapper_network": FixedStructureMapper15(c_dim=128),
            }
    cfg = OmegaConf.create(raw_cfg, flags={"allow_objects": True})
    n_loras = len(cfg.lora.keys())

    # Add LoRA layers from configuration
    cfg_mask = add_lora_from_config(model, cfg, device=args.device)

    # -- Setup Training ------------------------------------------- #+

    # Define params to optimize
    mappers_params = list(
        filter(lambda p: p.requires_grad, reduce(lambda x, y: x + list(y.parameters()), model.mappers, []))
    )
    encoder_params = list(
        filter(lambda p: p.requires_grad, reduce(lambda x, y: x + list(y.parameters()), model.encoders, []))
    )
    logger.info(f"Number params Mapper Network(s) {sum(p.numel() for p in mappers_params):,}")
    logger.info(f"Number params Encoder Network(s) {sum(p.numel() for p in encoder_params):,}")
    logger.info(f"Number params all LoRAs(s) {sum(p.numel() for p in model.params_to_optimize):,}")

    optimizer = torch.optim.AdamW(model.params_to_optimize + mappers_params + encoder_params, lr=1e-4)
    lr_scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=0, num_training_steps=args.max_epochs * len(train_dataloader))

    # Grab a fixed validation batch for logging
    # val_batch = next(iter(val_dataloader))
    eval_batch = torch.load("/BS/optdif/work/data/ffhq/eval_batch/size_512.pt")

    # Prepare network
    logger.info("prepare network")
    prepared = accelerator.prepare(
        *model.mappers,
        *model.encoders,
        model.unet,
        optimizer,
        train_dataloader,
        val_dataloader,
        lr_scheduler,
    )
    mappers = prepared[: len(model.mappers)]
    encoders = prepared[len(model.mappers) : len(model.mappers) + len(model.encoders)]
    (unet, optimizer, train_dataloader, val_dataloader, lr_scheduler) = prepared[
        len(model.mappers) + len(model.encoders) :
    ]
    model.unet = unet
    model.mappers = mappers
    model.encoders = encoders

    # -- Training Loop -------------------------------------------- #
    logger.info("start training")
    global_step = 0
    for epoch in range(-1, args.max_epochs):
        logger.info(f"Epoch {epoch}/{args.max_epochs}")
        unet.train()
        map(lambda m: m.train(), model.mappers)
        map(lambda e: e.train(), model.encoders)

        # Training steps
        for step, batch in enumerate(train_dataloader):

            # Log initial validation results before training starts
            if epoch == -1:
                break

            with accelerator.accumulate(unet, *mappers, *encoders):
                imgs = batch.to(accelerator.device)
                cs = [imgs] * n_loras
                prompts = [""] * batch.size(0)

                # cfg mask to always true such that the model always learns dropout
                model_pred, loss, x0, _ = model.forward_easy(
                    imgs,
                    prompts,
                    cs,
                    cfg_mask=cfg_mask,
                    batch=batch,
                )

                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
 
            if step % log_every_n_steps == 0:
                accelerator.log(
                    values={
                        "epoch": epoch,
                        "train/loss": loss.detach().item(),
                        "train/lr": lr_scheduler.get_last_lr()[0]
                    },
                    step=global_step
                )
            
            # after every gradient update step
            if accelerator.sync_gradients:
                global_step += 1

        # Validation steps
        with torch.no_grad():
            unet.eval()
            map(lambda m: m.eval(), mappers)
            map(lambda m: m.eval(), encoders)

            val_loss = 0.0
            val_steps = 0
            # -- Validation loss ---------------------------------- #
            for batch in val_dataloader:

                imgs = batch.to(accelerator.device)
                cs = [imgs] * n_loras
                prompts = [""] * batch.size(0)

                _, loss, _, _ = model.forward_easy(
                    imgs,
                    prompts,
                    cs,
                    cfg_mask=cfg_mask,
                    batch=batch,
                )

                val_loss += loss.detach().item()
                val_steps += 1

            val_loss /= val_steps
            accelerator.log(
                values={
                    "val/loss": val_loss,
                    "val/lr": lr_scheduler.get_last_lr()[0]
                },
                step=global_step
            )

            # -- FID and Spectral scores -------------------------- #
            val_recons_imgs_local = []
            for batch in val_dataloader:

                imgs = batch.to(accelerator.device)
                cs = [imgs] * n_loras
                prompts = [""] * batch.size(0)

                preds = model.sample_custom(
                    prompt=prompts,
                    num_images_per_prompt=1,
                    cs=cs,
                    generator=Generator(device="cuda").manual_seed(42),
                    cfg_mask=cfg_mask,
                )

                val_recons_imgs_local.append(
                    torch.stack([TF.to_tensor(img).to(accelerator.device) for img in preds])
                )

            val_recons_imgs_local = torch.cat(val_recons_imgs_local, dim=0)
            val_recons_imgs = accelerator.gather_for_metrics(val_recons_imgs_local).float().cpu()

            if accelerator.is_main_process:
                # Compute FID
                fid_score = fid_instance.compute_score_from_data(val_recons_imgs)
                logger.info(f"FID Score: {fid_score:.4f}")
                
                # Compute Spectral score
                spectral_score = spectral_instance.compute_score_from_data(val_recons_imgs)
                logger.info(f"Spectral Score: {spectral_score:.4f}")

                accelerator.log(
                    values={
                        "val/fid_score": fid_score,
                        "val/spectral_score": spectral_score,
                    },
                    step=global_step,
                )

            # -- Log reconstructions ------------------------------ #
            imgs = eval_batch.to(accelerator.device)
            cs = [imgs] * n_loras
            prompts = [""] * eval_batch.size(0)

            preds = model.sample_custom(
                prompt=prompts,
                num_images_per_prompt=1,
                cs=cs,
                generator=Generator(device="cuda").manual_seed(42),
                cfg_mask=cfg_mask,
            )
            
            # Log sampled images
            if accelerator.is_main_process:
                log_cond = np.asarray(imgs.cpu())
                log_cond = einops.rearrange(log_cond, "b c h w -> b h w c")
                log_cond = (log_cond + 1.0) / 2.0

                log_pred = np.stack(
                    [np.asarray(img.resize((512, 512))) for img in preds],
                    axis=0,
                )
                log_pred = log_pred / 255.0

                # Build an image grid
                cond  = torch.from_numpy(log_cond).permute(0,3,1,2)
                pred  = torch.from_numpy(log_pred).permute(0,3,1,2)
                panel = torch.cat([cond, pred], dim=0)
                grid  = vutils.make_grid(panel, nrow=eval_batch.size(0), value_range=(0,1))
            
                for tracker in accelerator.trackers:
                    if tracker.name == "tensorboard":
                        tracker.writer.add_image("reconstructions", grid, global_step)
        
        # -- Save checkpoint -------------------------------------- #
        if accelerator.is_main_process and epoch > -1:
            logger.info("Saving checkpoint")
            save_checkpoint(
                unet_sds=model.get_lora_state_dict(accelerator.unwrap_model(unet)),
                mapper_network_sd=[accelerator.unwrap_model(m).state_dict() for m in mappers],
                encoder_sd=None,
                path=output_dir / f"checkpoints/epoch_{epoch:03d}",
            )