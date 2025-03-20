import torch
from torch.utils import data
from PIL import Image
from torchvision import transforms

import os
import json
from tqdm import tqdm


class FFHQ(data.Dataset):
    """Dataset class for the FFHQ dataset."""

    def __init__(self, img_dir, pt_dir, attr_path, max_property_value=5, min_property_value=0, do_preprocess=False):
        """Initialize and preprocess the FFHQ dataset."""
        self.img_dir = img_dir
        self.pt_dir = pt_dir
        self.attr_path = attr_path
        self.dataset = []
        self.max_property_value = max_property_value
        self.min_property_value = min_property_value
        self.pt_dir = pt_dir

        # Load the attribute JSON file
        with open(self.attr_path, 'r') as f:
            attr_dict = json.load(f)

        # Fill dataset with sorted filenames and attribute data
        for key in sorted(attr_dict.keys()):
            filename = key.split('.')[0]
            if attr_dict[key] <= self.max_property_value and attr_dict[key] >= self.min_property_value:
                self.dataset.append([filename, attr_dict[key]])

        # Set the number of images
        self.num_images = len(self.dataset)

        # Check if all images have corresponding .pt files
        if not all([os.path.exists(f"{self.pt_dir}/{filename}.pt") for filename, _ in self.dataset]):
            do_preprocess = True

        if do_preprocess:
            self._preprocess()


    def _preprocess(self):
        """Preprocess the FFHQ dataset."""

        print("Preprocessing the FFHQ dataset...")

        # Define preprocessing transformations
        transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize to 256x256
            transforms.ToTensor(),         # Convert to tensor
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
        ])

        # Create directory for .pt files
        os.makedirs(self.pt_dir, exist_ok=True)
        
        # Convert images to .pt format
        for filename, _ in tqdm(self.dataset):
            # Load image
            image_path = os.path.join(self.img_dir, f"{filename}.png")
            image = Image.open(image_path).convert("RGB")

            # Apply transformations
            tensor = transform(image)

            # Save tensor to .pt file
            torch.save(tensor, os.path.join(self.pt_dir, f"{filename}.pt"))

        print("Conversion to .pt format complete. Finished preprocessing the FFHQ dataset.")


    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""

        # Get the image filename and attribute value
        filename, attr_value = self.dataset[index]

        # Load image
        image = torch.load(f"{self.data_dir}/{filename}.pt")

        return image.squeeze(0), attr_value

    def __len__(self):
        """Return the number of images."""
        return self.num_images