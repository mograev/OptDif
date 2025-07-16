"""
Weighted DataModule for the FFHQ dataset
"""

import json

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
import pytorch_lightning as pl
import numpy as np

from src.dataloader.utils import MultiModeDataset, OptEncodeDataset


class FFHQDataset(pl.LightningDataModule):
    """DataModule class for the FFHQ dataset with weighted sampling with support for multiple tensor modes."""

    def __init__(self, args, data_weighter=None, encoder=None, transform=None):
        """
        Initialize the FFHQDataset class.
        Args:
            args (argparse.Namespace): Command line arguments.
            data_weighter (object): DataWeighter object for weighting the dataset.
            encoder (object): Encoder object for encoding images.
            transform (callable, optional): Transform to apply to the images.
        """

        super().__init__()

        # Base directory path
        self.img_dir = args.img_dir

        # Dataset configuration
        self.attr_path = args.attr_path
        self.max_property_value = args.max_property_value
        self.min_property_value = args.min_property_value
        
        # DataLoader configuration
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.data_weighter = data_weighter
        self.val_split = args.val_split

        # Will be set in setup()
        self.data_train = None
        self.data_val = None
        self.attr_train = None
        self.attr_val = None
        self.train_dataset = None
        self.val_dataset = None

        # Transform to apply to the images
        self.transform = transform

        # Encoder for encoding images
        self.encoder = encoder
        # Device for the encoder
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Setup the dataset
        self._setup()

    @staticmethod
    def add_data_args(parent_parser):
        """Add data-related arguments to the argument parser."""
        data_group = parent_parser.add_argument_group(title="data")
        data_group.add_argument("--img_dir", type=str, required=True)
        data_group.add_argument("--attr_path", type=str, required=True)
        data_group.add_argument("--max_property_value", type=float, default=5.)
        data_group.add_argument("--min_property_value", type=float, default=0.)
        data_group.add_argument("--batch_size", type=int, default=128)
        data_group.add_argument("--num_workers", type=int, default=4)
        data_group.add_argument("--val_split", type=float, default=0.)

        return parent_parser

    def prepare_data(self):
        """Data preparation for the datamodule."""
        # No preparation needed here, keeping for consistency with LightningDataModule
        pass

    def _setup(self, stage=None):
        """Set up the dataset for training and validation."""
        # Load the attribute JSON file
        with open(self.attr_path, 'r') as f:
            attr_dict = json.load(f)
            
        # Fill dataset with sorted filenames and attribute data
        dataset = []
        for key in sorted(attr_dict.keys()):
            filename = key.split('.')[0]
            if attr_dict[key] >= self.min_property_value and attr_dict[key] < self.max_property_value:
                dataset.append([filename, attr_dict[key]])
        
        # Convert dataset to numpy array
        dataset_as_numpy = np.array(dataset)

        if self.val_split == 0.:
            self.data_train = dataset_as_numpy[:, 0].tolist()
            self.attr_train = dataset_as_numpy[:, 1].astype(np.float32)
            # Add pseudo validation batch for PyTorch Lightning
            self.data_val = dataset_as_numpy[0:self.batch_size, 0].tolist()
            self.attr_val = dataset_as_numpy[0:self.batch_size, 1].astype(np.float32)

        else:
            # Split the dataset into training and validation sets
            split_idx = int(len(dataset) * (1 - self.val_split))
            self.data_train = dataset_as_numpy[:split_idx, 0].tolist()
            self.attr_train = dataset_as_numpy[:split_idx, 1].astype(np.float32)
            self.data_val = dataset_as_numpy[split_idx:, 0].tolist()
            self.attr_val = dataset_as_numpy[split_idx:, 1].astype(np.float32)
            
        # Create tensor datasets
        self.train_dataset = OptEncodeDataset(
            filename_list=self.data_train,
            img_dir=self.img_dir,
            transform=self.transform,
            encoder=self.encoder,
            device=self.device
        )
        self.val_dataset = OptEncodeDataset(
            filename_list=self.data_val,
            img_dir=self.img_dir,
            transform=self.transform,
            encoder=self.encoder,
            device=self.device
        )

        # Set weights for sampling
        self.set_weights()


    def set_weights(self):
        """Set the weights for the weighted sampler."""

        if self.data_weighter is not None:
            # Set weights for training dataset
            self.train_weights = self.data_weighter.weighting_function(self.attr_train)
            self.train_sampler = WeightedRandomSampler(self.train_weights, len(self.train_weights), replacement=True)

            # Set weights for validation dataset
            self.val_weights = self.data_weighter.weighting_function(self.attr_val)
            self.val_sampler = WeightedRandomSampler(self.val_weights, len(self.val_weights), replacement=True)

        else:
            # If no data weighter is provided, use uniform sampling
            self.train_sampler = None
            self.val_sampler = None

    def set_encode(self, do_encode):
        """
        Set whether to encode images or not.
        Args:
            do_encode (bool): Whether to encode images.
        Returns:
            self: Returns the updated FFHQWeightedDataset instance.
        """
        # Update the encoder setting
        self.train_dataset.set_encode(do_encode)
        self.val_dataset.set_encode(do_encode)
        
        return self
    
    def append_train_data(self, data, labels):
        """
        Append data to the training dataset.
        Args:
            data (list): List of filenames or tensors to append.
            labels (numpy.ndarray): Corresponding labels for the data.
        Raises:
            AssertionError: If the length of data and labels do not match.
        """
        # Check if data is a list of filenames or a tensor
        self.data_train += data
        self.attr_train = np.append(self.attr_train, labels, axis=0)
        assert len(self.data_train) == len(self.attr_train), "Data and labels must have the same length."

        # Create tensor dataset
        self.train_dataset = OptEncodeDataset(
            filename_list=self.data_train,
            img_dir=self.img_dir,
            transform=self.transform,
            encoder=self.encoder,
            device=self.device
        )

        # Set weights
        self.set_weights()

    def train_dataloader(self):
        """Return the training DataLoader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.train_sampler,
            drop_last=False,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        """Return the validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.val_sampler,
            drop_last=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )