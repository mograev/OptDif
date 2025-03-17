import torch
import torchvision.transforms as transforms
import numpy as np

from PIL import Image

import json

from src.resnet50 import resnet50

# test plotting
import matplotlib.pyplot as plt
import os


class SmileClassifier:
    def __init__(self, model_path, attr_file):
        self.model = resnet50(attr_file=attr_file)
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model.eval()  # Set to evaluation mode

        # Load attribute mapping from JSON file
        with open(attr_file, 'r') as f:
            self.attr_data = json.load(f)

        # Get the new index for the old smile index 32
        self.smile_index = "32"
        self.smile_index_new = self.attr_data["attrIdx_to_newIdx"][self.smile_index]

        # extract smile classes
        self.smile_classes = self.attr_data["attr_info"][self.smile_index]["value"]
        self.smile_classes = np.array(self.smile_classes)

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # ResNet expects 224x224 images
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Standard ImageNet normalization
        ])

    def classify(self, image_path):
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0)

        # Run classification
        with torch.no_grad():
            output = self.model(input_tensor)

        # Ensure output is a tensor
        if isinstance(output, list):
            output = torch.cat(output, dim=0)

        # Convert output to probabilities using softmax
        probabilities = torch.nn.functional.softmax(output, dim=1).cpu().numpy()

        # Compute probability weighted smile score
        smile_score = probabilities[self.smile_index_new] @ self.smile_classes

        return smile_score
    

if __name__ == "__main__":
    # Configuration
    PRETRAINED_CLASSIFIER_PATH = "models/classifier/celeba_smile/predictor_128.pth.tar"
    ATTR_FILE = "models/classifier/celeba_smile/attributes.json"
    IMAGE_PATH = "data/ffhq/images1024x1024/"

    # Create classifier
    smile_classifier = SmileClassifier(PRETRAINED_CLASSIFIER_PATH, ATTR_FILE)

    # Demo: Classify 9 random images and plot them in a 3x3 grid with their smile score
    fig, axs = plt.subplots(3, 3, figsize=(10, 10))
    for i in range(3):
        for j in range(3):
            image_path = np.random.choice(os.listdir(IMAGE_PATH))
            smile_score = smile_classifier.classify(IMAGE_PATH + image_path)
            image = Image.open(IMAGE_PATH + image_path)
            axs[i, j].imshow(image)
            axs[i, j].set_title(f"{image_path}: {smile_score:.2f}")
            axs[i, j].axis("off")

    plt.title("Smile Classification Demo")
    plt.tight_layout()
    plt.savefig("demo/results/smile_classification_test.png")