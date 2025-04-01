from src.models.latent_vae import LatentVAE

# Model configuration
ddconfig = {
    "double_z": True,           # Use double z for VAE
    "z_channels": 8,            # Channels in the bottleneck
    "resolution": 32,           # Input resolution (for latents)
    "in_channels": 16,          # Input channels (for latents)
    "out_ch": 16,               # Output channels
    "ch": 64,                   # Base channel count
    "ch_mult": [1, 2, 2, 2],    # Channel multiplier for each resolution
    "num_res_blocks": 2,        # ResNet blocks per resolution
    "attn_resolutions": [],     # Resolutions at which to apply attention
    "dropout": 0.0,             # Dropout rate
    "attn_type": "none"         # Type of attention ("vanilla", "linear", or "none")
}

# Loss configuration
lossconfig = {
    "target": "src.models.modules.losses.SimpleVAELoss",
    "params": {
        "beta": 0.01,        # Weight for the KL divergence term
    }
}

# Other configuration
embed_dim = 128
ckpt_path = None


# init the LatentVAE model
model = LatentVAE(
    ddconfig=ddconfig,
    lossconfig=lossconfig,
    embed_dim=embed_dim,
    ckpt_path=ckpt_path
)

# Load one example image
import torch
IMAGE_TENSOR_PATH = "data/ffhq/pt_images/54321.pt"

img_tensor = torch.load(IMAGE_TENSOR_PATH)

# Add batch dimension: Change from (C, H, W) to (B, C, H, W)
img_tensor = img_tensor.unsqueeze(0)

# Load Stable Diffusion VAE model
from diffusers import AutoencoderKL

sd_vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-3.5-medium", subfolder="vae")
sd_vae.eval()

# Encode the image using the Stable Diffusion VAE
sd_latent = sd_vae.encode(img_tensor).latent_dist.sample()
sd_latent = sd_latent * sd_vae.config.scaling_factor

print("SD Latent shape:", sd_latent.shape)

# Encode the latent using the LatentVAE
latent = model.encode(sd_latent).sample()

print("Latent shape:", latent.shape)