"""
Data module for the ImageNet dataset.
"""

import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl

from src.dataloader.utils import ImageFolderToLatentDataset

class ImageNetDataset(pl.LightningDataModule):
    def __init__(self, args, encoder):
        super().__init__()
        self.img_dir = args.img_dir
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.device = args.data_device
        self.encoder = encoder

        self.aug = args.aug
        if self.aug:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            ])
        else:
            self.transform = None

    @staticmethod
    def add_data_args(parent_parser):
        """Add data-related arguments to the argument parser."""
        data_group = parent_parser.add_argument_group(title="data")
        data_group.add_argument("--img_dir", type=str, required=True)
        data_group.add_argument("--batch_size", type=int, default=128)
        data_group.add_argument("--num_workers", type=int, default=4)
        data_group.add_argument("--data_device", type=str, default='cpu')
        data_group.add_argument("--aug", action='store_true', default=False)

        return parent_parser

    def setup(self, stage=None):
        # Image directories
        train_dir = os.path.join(self.img_dir, "train")
        val_dir = os.path.join(self.img_dir, "val")

        # Setup datasets
        self.train_dataset = ImageFolderToLatentDataset(
            img_dir=train_dir,
            encoder=self.encoder,
            device=self.device,
            transform=self.transform
        )
        self.val_dataset = ImageFolderToLatentDataset(
            img_dir=val_dir,
            encoder=self.encoder,
            device=self.device,
            transform=self.transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            shuffle=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            shuffle=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )