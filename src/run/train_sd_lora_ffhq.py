import argparse
from omegaconf import OmegaConf
from functools import reduce

import numpy as np
import torch
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.transforms.functional as TF
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers.optimization import get_scheduler
import einops

from src.dataloader.ffhq import FFHQWeightedDataset
from src.dataloader.weighting import DataWeighter
from src.ctrloralter.model import SD15
from src.ctrloralter.annotators.openclip import VisionModel
from src.ctrloralter.mapper_network import SimpleMapper
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
    parser = FFHQWeightedDataset.add_data_args(parser)
    parser = DataWeighter.add_weight_args(parser)

    # Direct arguments
    parser.add_argument("--model_version", type=str, default="v1", help="Version of the model to use")
    parser.add_argument("--model_output_dir", type=str, help="Directory to save the model")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of epochs to train the model")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training (e.g., 'cuda' or 'cpu')")
    parser.add_argument("--num_devices", type=int, default=4, help="Number of devices to use for training")
    args = parser.parse_args()

    # Seed everything
    pl.seed_everything(42)

    # Setup logger
    logger = get_logger("ctrloralter_ffhq")

    # -- Load Data module ----------------------------------------- #

    # Load data
    datamodule = FFHQWeightedDataset(
        args,
        DataWeighter(args),
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop(size=(512, 512), scale=(0.9, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]),
    )
    datamodule.set_mode("img")

    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()

    # -- Setup Accelerator ---------------------------------------- #

    accelerator = Accelerator(
        project_dir=args.model_output_dir,
        log_with="tensorboard",
        gradient_accumulation_steps=1,
        mixed_precision="bf16",
    )

    # Init trackers
    logger.info("init trackers")
    if accelerator.is_main_process:
        accelerator.init_trackers("")

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
            }
        },
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
    lr_scheduler = get_scheduler("constant", optimizer=optimizer)

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
    for epoch in range(args.max_epochs):
        logger.info(f"Epoch {epoch + 1}/{args.max_epochs}")
        unet.train()
        map(lambda m: m.train(), model.mappers)
        map(lambda e: e.train(), model.encoders)

        # Training steps
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet, *mappers, *encoders):
                imgs = batch.to(accelerator.device)
                cs = [imgs] * n_loras
                prompts = [""] * batch.size(0)

                # cfg mask to always true such that the model always learns dropout
                model_pred, loss, x0, _ = model.forward_easy(
                    imgs,
                    prompts,
                    cs,
                    cfg_mask=[True for _ in cfg_mask],
                    batch=batch,
                )

                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
 
            accelerator.log(
                values={
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
            # Compute validation loss
            for i, val_batch in enumerate(val_dataloader):

                imgs = val_batch.to(accelerator.device)
                cs = [imgs] * n_loras
                prompts = [""] * batch.size(0)

                _, loss, _, _ = model.forward_easy(
                    imgs,
                    prompts,
                    cs,
                    cfg_mask=[True for _ in cfg_mask],
                    batch=val_batch,
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

            # Use first batch for image logging
            for i, val_batch in enumerate(val_dataloader):
                if i > 0:
                    break
                imgs = val_batch.to(accelerator.device)
                cs = [imgs] * n_loras
                prompts = [""] * batch.size(0)

                preds = model.sample_custom(
                    prompt=prompts,
                    num_images_per_prompt=1,
                    cs=cs,
                    cfg_mask=cfg_mask,
                )
            
            # Log sampled images
            if accelerator.is_main_process:
                lp = imgs.cpu()
                lp = torch.nn.functional.interpolate(
                    lp,
                    size=(512, 512),
                    mode="bicubic",
                    align_corners=False,
                )
                log_cond = TF.to_pil_image(einops.rearrange(lp, "b c h w -> c h (b w) "))
                log_cond = log_cond.convert("RGB")
                log_cond = np.asarray(log_cond)

                log_pred = np.concatenate(
                    [np.asarray(img.resize((cfg.size, cfg.size))) for img in preds],
                    axis=1,
                )
            
                for tracker in accelerator.trackers:
                    if tracker.name == "tensorboard":
                        np_images = np.concatenate([log_cond, log_pred], axis=0)
                        tracker.writer.add_images(
                            "validation",
                            np_images,
                            global_step=global_step,
                            dataformats="HWC",
                        )

        # Save checkpoint
        if accelerator.is_main_process:
            logger.info("Saving checkpoint")
            save_checkpoint(
                unet_sds=model.unet.lora_state_dict(accelerator.unwrap_model(unet)),
                mapper_network_sd=[accelerator.unwrap_model(m).state_dict() for m in mappers],
                encoder_sd=None,
                path=args.model_output_dir / f"checkpoints/epoch_{epoch:03d}",
            )