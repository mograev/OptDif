import os
import json
from tqdm import tqdm
import torch
from src.classification.smile_classifier import SmileClassifier

# Configuration
PRETRAINED_CLASSIFIER_PATH = "models/classifier/celeba_smile/predictor_128.pth.tar"
SCALED_CLASSIFIER_PATH = "models/classifier/celeba_smile/predictor_128_scaled3.pth.tar"
ATTR_FILE = "models/classifier/celeba_smile/attributes.json"
IMAGE_PATH = "data/ffhq/images1024x1024/"
OUTPUT_FILE = "data/ffhq/ffhq_smile_scores_scaled.json"

# Create classifier
device = "cuda" if torch.cuda.is_available() else "cpu"
smile_classifier = SmileClassifier(PRETRAINED_CLASSIFIER_PATH, ATTR_FILE, SCALED_CLASSIFIER_PATH, device=device)

# Get all images in the dataset directory
print("Classifying images...")
filenames = os.listdir(IMAGE_PATH)
# Ensure it's an image file
image_filenames = [filename for filename in filenames if filename.lower().endswith(('.png', '.jpg', '.jpeg'))]
# Sort the filenames so that the order is consistent
image_filenames.sort()
# Get the full filepaths
image_filepaths = [os.path.join(IMAGE_PATH, filename) for filename in image_filenames]

# Classify the images
smile_scores = smile_classifier.classify(image_filepaths)

# Create a dictionary of image filenames and their corresponding smile scores
smile_scores = {filename: smile_score for filename, smile_score in zip(image_filenames, smile_scores)}

# Save results to a JSON file
with open(OUTPUT_FILE, "w") as f:
    json.dump(smile_scores, f, indent=4)

print(f"Classification complete. Results saved to {OUTPUT_FILE}.")