"""
OpenCLIP vision encoder wrapper.
Loads pretrained vision tower and returns normalized image features for conditioning.
Source: https://github.com/CompVis/LoRAdapter/blob/main/src/annotators/openclip.py
"""

from torch import nn
import open_clip
import torch
from transformers import CLIPImageProcessor
from torchvision.transforms.functional import normalize
from .util import better_resize


class VisionModel(nn.Module):

    def __init__(
        self,
        clip_model: str,
        local_files_only: bool = True,
    ) -> None:
        super().__init__()
        self.clip_model = clip_model
        self.clip_vision_model = open_clip.create_model_from_pretrained("hf-hub:" + clip_model, return_transform=False)
        self.image_size = 224

        self.clip_vision_model.requires_grad_(False)
        self.clip_vision_model.eval()

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        assert imgs.min() >= -1.0
        assert imgs.max() <= 1.0
        assert len(imgs.shape) == 4

        imgs = (imgs + 1.0) / 2.0
        imgs = better_resize(imgs, self.image_size)
        imgs = normalize(
            imgs,
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )

        image_features = self.clip_vision_model.encode_image(imgs)

        return image_features