""" Encode FFHQ images to SD latents. """

import os
import pickle
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from diffusers import AutoencoderKL

from src.dataloader.utils import SimpleFilenameToTensorDataset


# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dir = "data/ffhq/pt_images"
output_dir = "data/ffhq/sd_latents"
batch_size=128
num_workers=4

print(f"Working on device: {device}")

# Load the file names to process
filename_list = os.listdir(input_dir)
# Ensure it's an tensor file and remove the extension
filename_list = [filename.split(".")[0] for filename in filename_list if filename.lower().endswith('.pt')]
# Sort the filenames so that the order is consistent
filename_list.sort()

# Create a DataLoader
dataset = SimpleFilenameToTensorDataset(
    filename_list=filename_list,
    tensor_dir=input_dir,
)
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=False,
)

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
    for image_tensor_batch in tqdm(dataloader):
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
for filename, latent in zip(filename_list, z_encode):    
    # Save the latent tensor
    torch.save(latent, os.path.join(output_dir, f"{filename}.pt"), pickle_protocol=pickle.HIGHEST_PROTOCOL)