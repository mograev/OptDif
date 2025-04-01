import torch
from torch.utils.data import Dataset

import os


class SimpleFilenameToTensorDataset(Dataset):
    """ Implements a dataset that transforms filenames to corresponding tensors """
    def __init__(self, filename_list, tensor_dir):
        self.filename_list = filename_list
        self.tensor_dir = tensor_dir

    def __getitem__(self, index):
        filename = self.filename_list[index]
        
        # Check if filename already has an extension
        base_name = os.path.basename(filename)
        name_without_ext = os.path.splitext(base_name)[0]
        
        # Try loading the tensor file
        try:
            path = os.path.join(self.tensor_dir, f"{name_without_ext}.pt")
            tensor_file = torch.load(path, weights_only=False)
        except Exception as e:
            # Fallback to loading directly from filename
            tensor_file = torch.load(filename)
            
        # Return tensor
        return tensor_file

    def __len__(self):
        return len(self.filename_list)
    

class MultiModeDataset(Dataset):
    """ 
    Implements a dataset that can switch between different representations
    (e.g., images and latents) based on the same filenames
    """
    def __init__(self, filename_list, mode_dirs=None, default_mode='img'):
        """
        Args:
            filename_list: List of filenames (without extension)
            mode_dirs: Dict mapping modes to directories, e.g., {'image': '/path/to/images', 'latent': '/path/to/latents'}
            default_mode: Default mode to use ('image', 'latent', etc.)
        """
        self.filename_list = filename_list
        self.mode_dirs = mode_dirs or {}
        
        if not self.mode_dirs and default_mode != 'direct':
            raise ValueError("No mode directories provided. Set default_mode='direct' to load directly from filenames.")
        
        self.mode = default_mode

    def set_mode(self, mode):
        """
        Switch between different representation modes
        Args:
            mode: One of the keys in mode_dirs or 'direct' to use filenames directly
        """
        self.mode = mode
        return self
    
    def add_mode(self, name, dir):
        """
        Add a new mode to the dataset
        Args:
            name: Name of the new mode
            dir: Directory for the new mode
        """
        self.mode_dirs[name] = dir
        return self

    def __getitem__(self, index):
        filename = self.filename_list[index]

        # Get the directory for the current mode
        mode_dir = self.mode_dirs[self.mode]
        
        # Check if filename already has an extension
        base_name = os.path.basename(filename)
        name_without_ext = os.path.splitext(base_name)[0]
        
        # Try loading from the appropriate directory
        try:
            if self.mode == 'img':
                path = os.path.join(mode_dir, f"{name_without_ext}.png")
            elif self.mode == 'img_tensor' or self.mode == 'sd_latent':
                path = os.path.join(mode_dir, f"{name_without_ext}.pt")
            else:
                path = os.path.join(filename)

            # Try loading from the path
            file = torch.load(path, weights_only=False)

            return file
            
        except Exception as e:
            raise RuntimeError(f"Failed to load tensor for {filename} in mode {self.mode} from path {path}: {str(e)}")

    def __len__(self):
        return len(self.filename_list)
