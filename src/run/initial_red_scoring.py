import os
import json
from tqdm import tqdm
import torch
from src.metrics.red import calculate_red_percentage

# Configuration
IMAGE_TENSOR_PATH = "data/ffhq/pt_images/"
OUTPUT_FILE = "data/ffhq/ffhq_red_scores.json"

# Get all images in the dataset directory
print("Classifying images...")
filenames = os.listdir(IMAGE_TENSOR_PATH)
# Sort the filenames so that the order is consistent
filenames.sort()

# Initialize a list to store red scores
red_scores = []

# Iterate through each image tensor files
for filename in tqdm(filenames):
    # Load the image tensor
    image_path = os.path.join(IMAGE_TENSOR_PATH, filename)
    image_tensor = torch.load(image_path)

    # Calculate the red percentage
    red_score = calculate_red_percentage(image_tensor)

    # Append the red score to the list
    red_scores.append(red_score)

# Create a dictionary of image filenames and their corresponding red scores
red_scores = {filename: red_score for filename, red_score in zip(filenames, red_scores)}

# Save results to a JSON file
with open(OUTPUT_FILE, "w") as f:
    json.dump(red_scores, f, indent=4)

print(f"Classification complete. Results saved to {OUTPUT_FILE}.")