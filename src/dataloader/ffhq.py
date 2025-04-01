import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
import pytorch_lightning as pl
from PIL import Image
from torchvision import transforms
import numpy as np

import os
import json
from tqdm import tqdm

from src.dataloader.utils import SimpleFilenameToTensorDataset, MultiModeDataset


# class FFHQ(Dataset):
#     """Dataset class for the FFHQ dataset."""

#     def __init__(self, img_dir, img_tensor_dir, attr_path, max_property_value=5, min_property_value=0, do_preprocess=False):
#         """Initialize and preprocess the FFHQ dataset."""
#         self.img_dir = img_dir
#         self.img_tensor_dir = img_tensor_dir
#         self.attr_path = attr_path
#         self.dataset = []
#         self.max_property_value = max_property_value
#         self.min_property_value = min_property_value

#         # Load the attribute JSON file
#         with open(self.attr_path, 'r') as f:
#             attr_dict = json.load(f)
            
#         # Fill dataset with sorted filenames and attribute data
#         for key in sorted(attr_dict.keys()):
#             filename = key.split('.')[0]
#             if attr_dict[key] <= self.max_property_value and attr_dict[key] >= self.min_property_value:
#                 self.dataset.append([filename, attr_dict[key]])
        
#         # Set the number of images
#         self.num_images = len(self.dataset)

#         # Preprocess the dataset if necessary
#         if do_preprocess:
#             self._preprocess()


#     def _preprocess(self):
#         """Preprocess the FFHQ dataset."""

#         print("Preprocessing the FFHQ dataset...")
        
#         # Define preprocessing transformations
#         transform = transforms.Compose([
#             transforms.Resize((256, 256)),  # Resize to 256x256
#             transforms.ToTensor(),         # Convert to tensor
#             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
#         ])

#         # Create directory for tensor files
#         os.makedirs(self.img_tensor_dir, exist_ok=True)

#         # Get a list of all images that need to be converted
#         images_to_convert = [filename for filename, _ in self.dataset if not os.path.exists(f"{self.img_tensor_dir}/{filename}.pt")]
        
#         # Convert images to .pt format
#         for filename, _ in tqdm(images_to_convert):
#             # Load image
#             image_path = os.path.join(self.img_dir, f"{filename}.png")
#             image = Image.open(image_path).convert("RGB")

#             # Apply transformations
#             tensor = transform(image)

#             # Save tensor to .pt file
#             torch.save(tensor, os.path.join(self.img_tensor_dir, f"{filename}.pt"))

#         print("Conversion to .pt format complete. Finished preprocessing the FFHQ dataset.")


#     def __getitem__(self, index):
#         """Return one image and its corresponding attribute label."""

#         # Get the image filename and attribute value
#         filename, attr_value = self.dataset[index]

#         # Load image
#         image = torch.load(f"{self.img_tensor_dir}/{filename}.pt")

#         return image.squeeze(0), attr_value

#     def __len__(self):
#         """Return the number of images."""
#         return self.num_images
    


class FFHQWeightedDataset(pl.LightningDataModule):
    """DataModule class for the FFHQ dataset with weighted sampling with support for multiple tensor modes."""

    def __init__(self, args, data_weighter):
        """Initialize the FFHQWeightedDataset class."""

        super().__init__()

        # Base directory paths
        self.img_dir = args.img_dir
        self.image_tensor_dir = args.img_tensor_dir
        self.mode_dirs = {
            'img': args.img_dir,
            'img_tensor': args.img_tensor_dir,
        }
        self.default_mode = 'img'

        # Dataset configuration
        self.attr_path = args.attr_path
        self.max_property_value = args.max_property_value
        self.min_property_value = args.min_property_value
        
        # DataLoader configuration
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.data_weighter = data_weighter

        # Will be set in setup()
        self.data_train = None
        self.data_val = None
        self.attr_train = None
        self.attr_val = None
        self.train_dataset = None
        self.val_dataset = None
        
        # Current mode for accessing data
        self.mode = self.default_mode

        # Setup the dataset
        self._setup()


    @staticmethod
    def add_data_args(parent_parser):
        data_group = parent_parser.add_argument_group(title="data")
        data_group.add_argument("--img_dir", type=str, required=True)
        data_group.add_argument("--img_tensor_dir", type=str, required=True)
        data_group.add_argument("--attr_path", type=str, required=True)
        data_group.add_argument("--max_property_value", type=float, default=5.)
        data_group.add_argument("--min_property_value", type=float, default=0.)
        data_group.add_argument("--batch_size", type=int, default=128)
        data_group.add_argument("--num_workers", type=int, default=4)

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
            if attr_dict[key] <= self.max_property_value and attr_dict[key] >= self.min_property_value:
                dataset.append([filename, attr_dict[key]])
        
        # Convert dataset to numpy array
        dataset_as_numpy = np.array(dataset)
        self.data_train = dataset_as_numpy[:, 0].tolist()
        self.attr_train = dataset_as_numpy[:, 1].astype(np.float32)

        # Add pseudo validation batch for PyTorch Lightning
        self.data_val = dataset_as_numpy[0:self.batch_size, 0].tolist()
        self.attr_val = dataset_as_numpy[0:self.batch_size, 1].astype(np.float32)
        
        # Create tensor datasets
        self.train_dataset = MultiModeDataset(
            filename_list=self.data_train,
            mode_dirs=self.mode_dirs,
            default_mode=self.default_mode
        )
        self.val_dataset = MultiModeDataset(
            filename_list=self.data_val,
            mode_dirs=self.mode_dirs,
            default_mode=self.default_mode
        )
        
        # Set weights for sampling
        self.set_weights()


    def set_weights(self):
        """Set the weights for the weighted sampler."""

        self.train_weights = self.data_weighter.weighting_function(self.attr_train)
        self.train_sampler = WeightedRandomSampler(self.train_weights, len(self.train_weights), replacement=True)

        self.val_weights = self.data_weighter.weighting_function(self.attr_val)
        self.val_sampler = WeightedRandomSampler(self.val_weights, len(self.val_weights), replacement=True)


    def add_mode(self, name, dir):
        """Add a new tensor mode to the dataset."""

        if name in self.mode_dirs:
            raise ValueError(f"Mode {name} already exists. Choose a different name.")
        
        # Update tensor directories dictionary
        self.mode_dirs[name] = dir
        
        # Update existing datasets if they exist
        self.train_dataset.add_mode(name, dir)
        self.val_dataset.add_mode(name, dir)
            
        return self

    def set_mode(self, mode):
        """Switch between different representation modes."""

        if mode not in self.mode_dirs and mode != 'direct':
            raise ValueError(f"Mode {mode} not found in mode_dirs. Available modes: {list(self.mode_dirs.keys()) + ['direct']}")
        
        self.mode = mode
        self.train_dataset.set_mode(mode)
        self.val_dataset.set_mode(mode)
        
        return self
    
    def get_mode_dir(self, mode):
        """Get the directory for a specific mode."""
        if mode not in self.mode_dirs:
            raise ValueError(f"Mode {mode} not found in mode_dirs. Available modes: {list(self.mode_dirs.keys())}")
        
        return self.mode_dirs[mode]

    def append_train_data(self, data, labels):
        """Append data to the training dataset."""

        # Check if data is a list of filenames or a tensor
        self.data_train += data
        self.attr_train = np.append(self.attr_train, labels, axis=0)
        assert len(self.data_train) == len(self.attr_train), "Data and labels must have the same length."

        # Create tensor dataset
        self.train_dataset = MultiModeDataset(
            filename_list=self.data_train,
            mode_dirs=self.mode_dirs,
            default_mode=self.default_mode
        )

        # Set weights
        self.set_weights()

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.train_sampler,
            drop_last=False
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.val_sampler,
            drop_last=False
        )