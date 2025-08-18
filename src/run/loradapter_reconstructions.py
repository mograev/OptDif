"""
FFHQ reconstructions with SD1.5 LoRAdapters (style/depth/HED).
Saves orig/recon images and computes LPIPS & FID.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
from omegaconf import OmegaConf

import torch
from torch import Generator
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.transforms.functional import pil_to_tensor

from src.dataloader.ffhq import FFHQDataset
from src.metrics.fid import FIDScore

from src.ctrloralter.model import SD15
from src.ctrloralter.annotators.openclip import VisionModel
from src.ctrloralter.annotators.midas import DepthEstimator
from src.ctrloralter.annotators.hed import TorchHEDdetector
from src.ctrloralter.mapper_network import SimpleMapper, FixedStructureMapper15
from src.ctrloralter.utils import add_lora_from_config

from taming.modules.losses.lpips import LPIPS


class TensorImageDataset(Dataset):
    """A simple dataset that returns pre-computed image tensors."""
    def __init__(self, tensors: List[torch.Tensor]):
        self.tensors = tensors

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        return self.tensors[idx]


def build_sd15_with_loras(device: str, approach: str) -> Tuple[object, List[str]]:
    """
    Build SD15 + selected LoRA adapters according to the named approach.
    Returns (sd_model, adapter_names) and sets sd_model.cfg_mask for CFG usage.
    """
    # Base SD15 model
    sd_model = SD15(
        pipeline_type="diffusers.StableDiffusionPipeline",
        model_name="runwayml/stable-diffusion-v1-5",
        local_files_only=False,
    ).eval()
    sd_model.to(device)

    # All approaches are 512px for SD15
    sd_model.img_size = 512

    # Common blocks for config construction
    style_block_initial = {
        "enable": "always",
        "optimize": False,
        "ckpt_path": "src/ctrloralter/checkpoints/sd15-style-cross-160-h",
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

    depth_block_initial = {
        "enable": "always",
        "optimize": False,
        "ckpt_path": "src/ctrloralter/checkpoints/sd15-depth-128-only-res",
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

    hed_block_initial = {
        "enable": "always",
        "optimize": False,
        "ckpt_path": "src/ctrloralter/checkpoints/sd15-hed-128-only-res",
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

    # Finetuned blocks (paths only differ)
    style_block_finetuned = dict(style_block_initial)
    style_block_finetuned = style_block_finetuned.copy()
    style_block_finetuned["ckpt_path"] = "models/sd_lora/version_9/checkpoints/epoch_053"

    style_depth_blocks_finetuned = (
        dict(style_block_initial),
        dict(depth_block_initial),
    )
    style_depth_blocks_finetuned[0]["ckpt_path"] = "models/sd_lora/version_7/checkpoints/epoch_053"
    style_depth_blocks_finetuned = list(style_depth_blocks_finetuned)

    style_hed_blocks_finetuned = (
        dict(style_block_initial),
        dict(hed_block_initial),
    )
    style_hed_blocks_finetuned[0]["ckpt_path"] = "models/sd_lora/version_8/checkpoints/epoch_053"
    style_hed_blocks_finetuned = list(style_hed_blocks_finetuned)

    # Map approach to config
    if approach == "initial_style":
        lora_cfg = {"lora": {"style": style_block_initial}}
    elif approach == "initial_style_depth":
        lora_cfg = {"lora": {"style": style_block_initial, "struct": depth_block_initial}}
    elif approach == "initial_style_hed":
        lora_cfg = {"lora": {"style": style_block_initial, "struct": hed_block_initial}}
    elif approach == "finetuned_style":
        lora_cfg = {"lora": {"style": style_block_finetuned}}
    elif approach == "finetuned_style_depth":
        lora_cfg = {"lora": {"style": style_depth_blocks_finetuned[0], "struct": style_depth_blocks_finetuned[1]}}
    elif approach == "finetuned_style_hed":
        lora_cfg = {"lora": {"style": style_hed_blocks_finetuned[0], "struct": style_hed_blocks_finetuned[1]}}
    else:
        raise ValueError(f"Unknown approach: {approach}")

    raw_cfg = {
        "ckpt_path": "ctrloralter/checkpoints",
        "ignore_check": False,
        **lora_cfg,
    }

    cfg = OmegaConf.create(raw_cfg, flags={"allow_objects": True})

    # Attach LoRAs
    cfg_mask = add_lora_from_config(sd_model, cfg, device=device)
    sd_model.cfg_mask = cfg_mask

    adapters = list(lora_cfg["lora"].keys())
    return sd_model, adapters


def denorm01(x: torch.Tensor) -> torch.Tensor:
    """Map [-1,1] -> [0,1] for saving."""
    return (x * 0.5 + 0.5).clamp(0, 1)


def normm11(x: torch.Tensor) -> torch.Tensor:
    """Map [0,1] -> [-1,1] for LPIPS and consistent metrics."""
    return (x - 0.5) / 0.5


def reconstruct_with_model(sd_model, imgs: torch.Tensor, use_structure: bool) -> torch.Tensor:
    """Reconstruct a batch with LoRAdapters. Returns recon in [-1,1]."""
    with torch.no_grad():
        # `imgs` are already tensors in [-1,1]. The adapters expect tensors.
        # Provide same image to all enabled adapters (style, and optionally structure)
        cs = [imgs]
        if use_structure:
            cs.append(imgs)
        sampled_images = sd_model.sample_custom(
            prompt="",
            num_images_per_prompt=imgs.size(0),
            cs=cs,
            generator=Generator(device=imgs.device),
            cfg_mask=sd_model.cfg_mask,
        )
        # sample_custom returns a list of PIL images
        recon = torch.stack([pil_to_tensor(pil_img) for pil_img in sampled_images], dim=0).float() / 255.0
        recon = recon.to(imgs.device)
        # Ensure correct spatial size
        if recon.shape[-2:] != (sd_model.img_size, sd_model.img_size):
            recon = torch.nn.functional.interpolate(recon, size=(sd_model.img_size, sd_model.img_size), mode='bilinear', align_corners=False)
        return normm11(recon)


def evaluate_approach(
    approach_name: str,
    device: str,
    val_loader,
    fid_instance: FIDScore,
    out_root: Path,
) -> Tuple[float, float]:
    """
    For a given approach, reconstruct the validation set, save images, compute LPIPS and FID.
    Returns (avg_lpips, fid_score).
    """
    sd_model, adapters = build_sd15_with_loras(device, approach_name)
    use_structure = any(a != "style" for a in adapters)

    lpips_fn = LPIPS().to(device).eval()

    approach_dir = out_root / approach_name
    (approach_dir / "orig").mkdir(parents=True, exist_ok=True)
    (approach_dir / "recon").mkdir(parents=True, exist_ok=True)

    # Storage for FID on reconstructions
    recon_tensors: List[torch.Tensor] = []

    # Iterate
    idx_global = 0
    for batch in val_loader:
        imgs = batch.to(device)  # [-1,1]
        with torch.no_grad():
            recons = reconstruct_with_model(sd_model, imgs, use_structure)

        # Save originals and reconstructions to disk (as [0,1])
        imgs_vis = denorm01(imgs.detach().cpu())
        recons_vis = denorm01(recons.detach().cpu())
        bsz = imgs_vis.size(0)
        for i in range(bsz):
            save_image(imgs_vis[i], approach_dir / "orig" / f"{idx_global + i:06d}.png")
            save_image(recons_vis[i], approach_dir / "recon" / f"{idx_global + i:06d}.png")

        # LPIPS accumulation (mean over batch)
        # LPIPS expects [-1,1]
        lpips_batch = lpips_fn(imgs, recons).view(-1)
        if idx_global == 0:
            lpips_sum = lpips_batch.sum().detach()
            n_count = torch.tensor(float(bsz), device=device)
        else:
            lpips_sum += lpips_batch.sum().detach()
            n_count += float(bsz)

        # For FID, collect recon normalized like the real set (here also [-1,1])
        recon_tensors.extend(list(recons.detach().cpu()))

        idx_global += bsz

    avg_lpips = (lpips_sum / n_count).item()

    # Compute FID
    recon_ds = TensorImageDataset(recon_tensors)
    fid_score = fid_instance.compute_score_from_data(recon_ds)

    return avg_lpips, float(fid_score)


def main():
    # CONFIGURATION
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = "results/loradapter_recons"
    # FFHQ config
    img_dir = "data/ffhq/images1024x1024"
    attr_path = "data/ffhq/smile_scores.json"
    max_property_value = 5
    min_property_value = 0
    num_workers = 4
    batch_size = 64
    val_split = 0.1

    # Collect args in namespace
    args = argparse.Namespace(
        img_dir=img_dir,
        attr_path=attr_path,
        max_property_value=max_property_value,
        min_property_value=min_property_value,
        num_workers=num_workers,
        batch_size=batch_size,
        val_split=val_split,
    )

    # Build FFHQ module
    img_size = 512
    datamodule = FFHQDataset(
        args,
        transform=transforms.Compose([
            transforms.Resize(size=img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    ).set_encode(False)

    # Extract a random 1000 image subset from the validation set
    val_loader = datamodule.val_dataloader()
    val_subset = torch.utils.data.Subset(val_loader.dataset, torch.randint(0, len(val_loader.dataset), (1000,)))
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    # Load real FID stats
    fid_instance = FIDScore(img_size=img_size, device=device, batch_size=64, num_workers=0)
    fid_instance.load_real_stats(f"data/ffhq/inception_stats/size_{img_size}_smile_{min_property_value}_{max_property_value}.pt")

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    approaches = [
        "initial_style",
        "initial_style_depth",
        "initial_style_hed",
        "finetuned_style",
        "finetuned_style_depth",
        "finetuned_style_hed",
    ]

    print("\n===== LoRA Adapter Reconstruction Analysis on FFHQ (val) =====")
    print(f"Device: {device} | Output: {out_root.resolve()}")

    # store results in dict
    result_dict = {}

    for name in approaches:
        print(f"\n-- Evaluating: {name} --")
        avg_lpips, fid_score = evaluate_approach(
            approach_name=name,
            device=device,
            val_loader=val_loader,
            fid_instance=fid_instance,
            out_root=out_root,
        )
        print(f"{name}: LPIPS={avg_lpips:.8f}, FID={fid_score:.4f}")

        # store results in dict
        result_dict[name] = {
            "LPIPS": avg_lpips,
            "FID": fid_score
        }

    # Save results to a JSON file
    with open(out_root / "results_1.json", "w") as f:
        json.dump(result_dict, f, indent=4)


if __name__ == "__main__":

    # seed everything for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    main()