import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
import pytorch_lightning as pl
from PIL import Image
from torchvision import transforms
import numpy as np

import os
import json
from tqdm import tqdm


class FFHQ(Dataset):
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

        # Get a list of all images that need to be converted
        images_to_convert = [filename for filename, _ in self.dataset if not os.path.exists(f"{self.pt_dir}/{filename}.pt")]
        
        # Convert images to .pt format
        for filename, _ in tqdm(images_to_convert):
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
        image = torch.load(f"{self.pt_dir}/{filename}.pt")

        return image.squeeze(0), attr_value


    def __len__(self):
        """Return the number of images."""
        return self.num_images
    


class FFHQWeightedTensorDataset(pl.LightningDataModule):
    """DataModule class for the FFHQ dataset with weighted sampling."""

    def __init__(self, args, data_weighter):
        """Initialize the FFHQWeightedTensorDataset class."""

        super().__init__()

        self.img_dir = args.img_dir
        self.pt_dir = args.pt_dir
        self.train_attr_path = args.train_attr_path
        self.val_attr_path = args.val_attr_path
        self.combined_attr_path = args.combined_attr_path
        self.max_property_value = args.max_property_value
        self.min_property_value = args.min_property_value
        self.mode = args.mode
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.data_weighter = data_weighter

        self.train_dataset = None
        self.val_dataset = None
        self.combined_dataset = None 

    @staticmethod
    def add_data_args(parent_parser):
        data_group = parent_parser.add_argument_group(title="data")
        data_group.add_argument("--img_dir", type=str, required=True)
        data_group.add_argument("--pt_dir", type=str, required=True)
        data_group.add_argument("--train_attr_path", type=str, required=True)
        data_group.add_argument("--val_attr_path", type=str, required=True)
        data_group.add_argument("--combined_attr_path", type=str, required=True)
        data_group.add_argument("--max_property_value", type=int, default=5)
        data_group.add_argument("--min_property_value", type=int, default=0)
        data_group.add_argument("--mode", type=str, default="split", choices=["split", "all"])
        data_group.add_argument("--batch_size", type=int, default=128)
        data_group.add_argument("--num_workers", type=int, default=4)

        return parent_parser

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        """Set up the dataset for training and validation."""

        if self.mode == "split":
            # Load training and validation datasets
            self.train_dataset = FFHQ(self.img_dir, self.pt_dir, self.train_attr_path, self.max_property_value, self.min_property_value)
            self.val_dataset = FFHQ(self.img_dir, self.pt_dir, self.val_attr_path, self.max_property_value, self.min_property_value)

            # Convert datasets to numpy arrays
            train_dataset_as_numpy = np.array(self.train_dataset.dataset)
            val_dataset_as_numpy = np.array(self.val_dataset.dataset)
            self.data_train = train_dataset_as_numpy[:, 0].tolist()
            self.attr_train = train_dataset_as_numpy[:, 1].astype(np.float32)
            self.data_val = val_dataset_as_numpy[:, 0].tolist()
            self.attr_val = val_dataset_as_numpy[:, 1].astype(np.float32)

        elif self.mode == "all":
            # Load combined dataset
            self.combined_dataset = FFHQ(self.img_dir, self.pt_dir, self.combined_attr_path, self.max_property_value, self.min_property_value)

            # Convert dataset to numpy array
            combined_dataset_as_numpy = np.array(self.combined_dataset.dataset)
            self.data_train = combined_dataset_as_numpy[:, 0].tolist()
            self.attr_train = combined_dataset_as_numpy[:, 1].astype(np.float32)
            # Add pseudo validation batch for PyTorch Lightning
            self.data_val = combined_dataset_as_numpy[0:self.batch_size, 0].tolist()
            self.attr_val = combined_dataset_as_numpy[0:self.batch_size, 1].astype(np.float32)

        else:
            raise ValueError("Invalid mode. Choose 'split' or 'all'.")
        
        # Create tensor datasets
        self.train_dataset = SimpleFilenameToTensorDataset(self.data_train, self.pt_dir)
        self.val_dataset = SimpleFilenameToTensorDataset(self.data_val, self.pt_dir)

        # Set weights
        self.set_weights()


    def set_weights(self):
        """Set the weights for the weighted sampler."""

        self.train_weights = self.data_weighter.weighting_function(self.attr_train)
        self.train_sampler = WeightedRandomSampler(self.train_weights, len(self.train_weights), replacement=True)

        self.val_weights = self.data_weighter.weighting_function(self.attr_val)
        self.val_sampler = WeightedRandomSampler(self.val_weights, len(self.val_weights), replacement=True)


    def append_train_data(self, data, labels):
        """Append data to the training dataset."""
        self.data_train += data
        self.attr_train = np.append(self.attr_train, labels, axis=0)

        # Create tensor dataset
        self.train_dataset = SimpleFilenameToTensorDataset(self.data_train, self.pt_dir)

        # Set weights
        self.set_weights()

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.train_sampler,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.val_sampler,
            drop_last=True
        )


class SimpleFilenameToTensorDataset(Dataset):
    """ Implements a dataset that transforms filenames to corresponding tensors """
    def __init__(self, filename_list, pt_dir):
        self.filename_list = filename_list
        self.pt_dir = pt_dir

    def __getitem__(self, index):
        filename = self.filename_list[index]

        # Check if filename is from original training data or sampled
        is_orig_training_data = True
        try:
            int(filename.split('.')[0])
        except:
            is_orig_training_data = False

        if is_orig_training_data:
            # Get the image index
            filename_idx = int(filename.split('.')[0])
            
            # Load image
            image = torch.load(f"{self.pt_dir}/{filename}.pt").unsqueeze(0)
        else:
            # Load image
            image = torch.load(filename).unsqueeze(0)

        # Return image tensor
        return tuple(image)

    def __len__(self):
        return len(self.filename_list)