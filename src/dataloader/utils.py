"""
Utility classes for datasets that load images from filenames.
"""

import os
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


class OptEncodeDataset(Dataset):
    """
    Implements a dataset that can switch between directly loading images,
    transforming them, and encoding them into latents.
    """
    def __init__(self, filename_list, img_dir, transform=None, encoder=None, device='cpu'):
        """
        Args:
            filename_list: List of filenames (without extension)
            img_dir: Directory where image files are stored
            transform: Optional transformation to apply to the images
            encoder: Optional encoder to use for encoding images
            device: Device to load the encoder onto ('cpu' or 'cuda')
        """
        self.filename_list = filename_list
        self.img_dir = img_dir
        self.device = device
        self.transform = transform

        if encoder:
            self.do_encode = True
            self.encoder = encoder
            self.encoder.eval()
            self.encoder.to(self.device)
        else:
            self.do_encode = False
            self.encoder = None

    def set_encode(self, do_encode):
        """
        Set whether to encode images or not
        Args:
            do_encode: Boolean indicating whether to encode images
        """
        self.do_encode = do_encode
        return self

    def __getitem__(self, index):
        """
        Args:
            index: Index of the item to retrieve
        Returns:
            tensor: Loaded tensor corresponding to the filename
        """
        # Get the filename
        filename = self.filename_list[index]

        # Check if filename already has an extension
        base_name = os.path.basename(filename)
        name_without_ext = os.path.splitext(base_name)[0]

        try:
            # Try loading from the set directory
            path = os.path.join(self.img_dir, f"{name_without_ext}.png")
            file = Image.open(path).convert("RGB")
        except:
            # Fallback to loading directly from filename
            path = os.path.join(filename)
            file = Image.open(path).convert("RGB")

        # Apply transformation
        file = self.transform(file)

        # Optionally encode the image to a latent tensor
        if self.do_encode:
            with torch.no_grad():
                file = file.unsqueeze(0).to(self.device)
                file = self.encoder.encode(file).latent_dist.sample()
                file = file.squeeze(0).cpu()

        return file

    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.filename_list)


class ImageFolderToLatentDataset(Dataset):
    """
    Implements a dataset that loads latents from image folder
    """
    def __init__(
        self,
        img_dir,
        encoder,
        device='cpu',
        transform=None
    ):
        """
        Args:
            img_dir: Directory containing the images
            encoder: Encoder to use for encoding images
            device: Device to load the encoder onto ('cpu' or 'cuda')
            transform: Optional transformation to apply to the images
        """
        self.img_dir = img_dir
        self.encoder = encoder.eval().to(device)
        self.device = device
        self.transform = transform or transforms.ToTensor()

        # Setup image folder
        self.img_folder = ImageFolder(
            root=img_dir,
            transform=self.transform
        )

    def __getitem__(self, index):
        """
        Args:
            index: Index of the item to retrieve
        Returns:
            tensor: Loaded latent tensor corresponding to the filename
        """
        # Get the image
        img_tensor, _ = self.img_folder[index]

        # Move image tensor to the *same* device as the encoder to avoid device‑mismatch
        encoder_device = next(self.encoder.parameters()).device
        img_tensor = img_tensor.unsqueeze(0).to(encoder_device, non_blocking=True)

        # Encode the image to a latent tensor
        with torch.no_grad():
            latent_tensor = self.encoder.encode(img_tensor).latent_dist.sample()

        # Always move the latent back to CPU so that the DataLoader can share it safely
        latent_tensor = latent_tensor.squeeze(0).cpu()
        return latent_tensor

    def __len__(self):
        return len(self.img_folder)