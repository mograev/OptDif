"""
Script to get inception stats for the FFHQ dataset, possibly specific to a smile score range or image size.
It is used to compute the FID score for reconstructions (during latent model training) and newly generated images (during latent space optimization).
"""

from argparse import Namespace

from torchvision import transforms

from src.dataloader.ffhq import FFHQDataset
from src.metrics.fid import FIDScore

# -- Configuration ------------------------------------------------ #
IMG_SIZE = 512  # Image size
MIN_PROPERTY_VALUE = 2  # Minimum smile score
MAX_PROPERTY_VALUE = 5  # Maximum smile score

# -- Load FFHQ dataset -------------------------------------------- #

# Initialize FFHQ dataset using CLI arguments
ffhq_dataset = FFHQDataset(
    args=Namespace(
		img_dir="data/ffhq/images1024x1024",
		attr_path="data/ffhq/smile_scores.json",
		max_property_value=MAX_PROPERTY_VALUE,
		min_property_value=MIN_PROPERTY_VALUE,
		batch_size=16,
		num_workers=0,
		val_split=0,
	),
    transform=transforms.Compose([
		transforms.Resize((IMG_SIZE, IMG_SIZE)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
	]),
)

# Extract the dataset that can be fed into a dataloader
ffhq_dataset = ffhq_dataset.train_dataset

# -- Initialize & Fit FID instance -------------------------------- #

# Initialize FID instance
fid_instance = FIDScore(img_size=IMG_SIZE, device="cuda") # change to "cpu" if CUDA is not available

# Fit the FID instance to the real data
fid_instance.fit_real(ffhq_dataset)

# Save the fitted FID statistics
fid_instance.save_real_stats(f"data/ffhq/inception_stats/size_{IMG_SIZE}_smile_{MIN_PROPERTY_VALUE}_{MAX_PROPERTY_VALUE}.pt")