import os
import json
from tqdm import tqdm
from itertools import batched

import torch
import torchvision.transforms as transforms
from PIL import Image

from src.classification.smile_classifier import SmileClassifier

# -- Configuration ------------------------------------------------ #
CLASSIFIER_PATH = "models/classifier/celeba_smile/predictor_128_scaled3.pth.tar" # "models/classifier/celeba_smile/predictor_128.pth.tar" # or scaled: "models/classifier/celeba_smile/predictor_128_scaled3.pth.tar"
ATTR_FILE = "models/classifier/celeba_smile/attributes.json"
IMAGE_PATH = "/BS/databases/CelebA/img_align_celeba/" # "data/ffhq/images1024x1024/" # 
OUTPUT_FILE = "data/celeba/smile_scores_scaled.json" # or scaled: "data/ffhq/smile_scores_scaled.json"
BATCH_SIZE = 128

# -- Load classifier ---------------------------------------------- #
device = "cuda" if torch.cuda.is_available() else "cpu"
smile_classifier = SmileClassifier(CLASSIFIER_PATH, ATTR_FILE, scaled=True, device=device) # Change to True for scaled classifier

# -- Load images -------------------------------------------------- #

# Get all images in the dataset directory
filenames = os.listdir(IMAGE_PATH)
image_filenames = [filename for filename in filenames if filename.lower().endswith(('.png', '.jpg', '.jpeg'))]
image_filenames.sort()
image_filepaths = [os.path.join(IMAGE_PATH, filename) for filename in image_filenames]

# -- Classify images in batches ----------------------------------- #

smile_scores = {}

for path_batch in tqdm(batched(image_filepaths, BATCH_SIZE), desc="Processing batches"):
    # Ensure path_batch is a list
    path_batch = list(path_batch)

    # Load and preprocess image tensors
    input_tensors = []
    for path in path_batch:
        # Load image and convert to tensor
        img = Image.open(path).convert("RGB")
        img = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])(img)
        input_tensors.append(img)

    # Stack all image tensors into one large tensor
    input_tensor = torch.stack(input_tensors)

    # Classify the images
    smile_scores_batch = smile_classifier.classify(input_tensor)

    # Update the smile scores dictionary with the results
    smile_scores.update({os.path.basename(filename): smile_score for filename, smile_score in zip(path_batch, smile_scores_batch)})

# Sort the smile scores by filename
smile_scores = dict(sorted(smile_scores.items()))

# -- Save results ------------------------------------------------- #

# Save results to a JSON file
with open(OUTPUT_FILE, "w") as f:
    json.dump(smile_scores, f, indent=4)

print(f"Classification complete. Results saved to {OUTPUT_FILE}.")