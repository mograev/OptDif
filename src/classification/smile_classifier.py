import torch
import torchvision.transforms as transforms
import numpy as np

from PIL import Image

import json

from src.classification.resnet50 import resnet50
from src.classification.temperature_scaling import ModelWithTemperature

SMILE_ATTR_IDX_NEW = 3
SMILE_ATTR_IDX_OLD = 32

class SmileClassifier:
    def __init__(self, model_path, attr_file, scaled_model_path=None, device="cpu"):
        self.model = resnet50(attr_file=attr_file)
        checkpoint = torch.load(model_path, map_location=torch.device(device))
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model.eval()  # Set to evaluation mode

        # Load temperature scaling
        if scaled_model_path:
            self.checkpoint_scaled_model = torch.load(scaled_model_path, map_location=torch.device(device))
            self.model = ModelWithTemperature(self.model)
            self.model.load_state_dict(self.checkpoint_scaled_model, strict=True)
            self.model.eval()

        # Load attribute mapping from JSON file
        with open(attr_file, 'r') as f:
            self.attr_data = json.load(f)

        # Extract smile classes
        self.smile_classes = self.attr_data["attr_info"][f"{SMILE_ATTR_IDX_OLD}"]["value"]
        self.smile_classes = np.array(self.smile_classes)

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # ResNet expects 224x224 images
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Standard ImageNet normalization
        ])


    def classify(self, image_paths, batch_size=128):
        """
        Classify a single image or a batch of images
        """

        # Ensure image_paths is a list
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        # Process images in batches if necessary
        all_outputs = []
        for i in range(0, len(image_paths), batch_size):
            # Logging
            print(f"Processing batch {i // batch_size + 1}/{len(image_paths) // batch_size + 1}", flush=True)

            # Get the current batch
            batch = image_paths[i : i + batch_size]

            # Load and preprocess images
            images = [Image.open(image_path).convert("RGB") for image_path in batch]
            input_tensors = torch.stack([self.transform(image) for image in images])

            # Run classification
            with torch.no_grad():
                output = self.model(input_tensors)

            # Ensure output is a tensor
            if isinstance(output, list):
                output = torch.stack(output, dim=0)

            # Convert output to probabilities using softmax
            probabilities = torch.nn.functional.softmax(output, dim=2).cpu().numpy()

            # Compute probability weighted smile score
            smile_scores = probabilities[SMILE_ATTR_IDX_NEW] @ self.smile_classes
            all_outputs.extend(smile_scores.tolist())

        # Return a single value for single input or a list otherwise
        return all_outputs[0] if len(all_outputs) == 1 else all_outputs
