""" Encode FFHQ images to SD latents. """

import os
import pickle
from tqdm import tqdm
from PIL import Image
import glob

import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

from diffusers import AutoencoderKL

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dir = "data/ffhq/images1024x1024"
output_dir = "data/ffhq/sd_latents"
os.makedirs(output_dir, exist_ok=True)

print(f"Working on device: {device}")

# Load the file names to process
img_paths = glob.glob(f"{input_dir}/*.png")
img_paths.sort()

# Create a DataLoader
class ImageFolderDataset(Dataset):
    def __init__(self, paths, size=(256,256)):
        self.paths = paths
        self.tf = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32),
        ])
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.tf(img)

dataset = ImageFolderDataset(img_paths, size=(256,256))
loader  = DataLoader(dataset, batch_size=128, num_workers=8, pin_memory=True)

# Load pre-trained SD-VAE model
sd_vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-3.5-medium", subfolder="vae")
sd_vae.eval()
sd_vae.to(device)

# Freeze the SD-VAE model
for param in sd_vae.parameters():
    param.requires_grad = False

z_encode = []

# Encode images to SD latents
with torch.no_grad():
    for image_tensor_batch in tqdm(loader):
        # Move images to the correct device
        images = image_tensor_batch.to(device)

        # Encode images into latent space (using mean of the latent distribution)
        latents = sd_vae.encode(images).latent_dist.mean

        # Save latents to disk
        for latent in latents:
            z_encode.append(latent.cpu())

# Free up GPU memory
sd_vae = sd_vae.cpu()
torch.cuda.empty_cache()

# Save the latents to disk
for filepath, latent in zip(img_paths, z_encode):
    # Derive a base name without extension for saving
    base = os.path.splitext(os.path.basename(filepath))[0]
    # Save the latent tensor
    torch.save(latent, os.path.join(output_dir, f"{base}.pt"), pickle_protocol=pickle.HIGHEST_PROTOCOL)

# Save latents as a single file
latents = torch.stack(z_encode, dim=0)
torch.save(latents, "data/ffhq/sd_latents.pt", pickle_protocol=pickle.HIGHEST_PROTOCOL)