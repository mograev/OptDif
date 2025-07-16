"""
SmileClassifier Class.
This class uses a pre-trained ResNet50 model to classify smile attributes
"""
import json

import torch
import numpy as np
import torchvision.transforms as transforms

from src.classification.resnet50 import resnet50
from src.classification.temperature_scaling import ModelWithTemperature

SMILE_ATTR_IDX_NEW = 3
SMILE_ATTR_IDX_OLD = 32


class SmileClassifier:
    def __init__(self, model_path, attr_file, scaled=False, device="cpu"):
        """
        Initialize the SmileClassifier.
        Args:
            model_path (str): Path to the model checkpoint.
            attr_file (str): Path to the attribute file.
            scaled (bool): If True, load a scaled model.
            device (str): Device to use for computation ("cpu" or "cuda").
        """
        # Model initialization
        self.model = resnet50(attr_file=attr_file)
        self.device = device

        # Load model checkpoint
        if scaled:
            checkpoint = torch.load(model_path, map_location=torch.device(device))
            self.model = ModelWithTemperature(self.model)
            self.model.load_state_dict(checkpoint, strict=True)
        else:
            checkpoint = torch.load(model_path, map_location=torch.device(self.device))
            self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model.eval()
        self.model.to(self.device)

        # Load attribute mapping from JSON file
        with open(attr_file, 'r') as f:
            self.attr_data = json.load(f)

        # Extract smile classes
        self.smile_classes = self.attr_data["attr_info"][f"{SMILE_ATTR_IDX_OLD}"]["value"]
        self.smile_classes = np.array(self.smile_classes)

    def classify(self, images, batch_size=128, return_prob=False):
        """
        Classify a single image or a batch of images from tensors.
        Args:
            images (torch.Tensor): Image(s) to classify.
            batch_size (int): Batch size for processing.
            return_prob (bool): If True, return probabilities instead of smile scores.
        Returns:
            np.ndarray: Smile scores or probabilities for the input images.
        """
        # Catch numpy arrays and convert to torch tensors
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)

        # Add batch dimension if needed
        if images.dim() == 3:
            images = images.unsqueeze(0)

        # Split input into smaller batches
        outputs = []
        for start_idx in range(0, len(images), batch_size):
            batch = images[start_idx : start_idx + batch_size]
            # Resize as needed
            batch = torch.nn.functional.interpolate(
                batch, size=(224, 224), mode="bilinear", align_corners=False, antialias=True
            ).to(self.device)

            # Normalize the images
            batch = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(batch)

            # Forward pass
            with torch.no_grad():
                preds = self.model(batch)

            # Process outputs
            if isinstance(preds, list):
                preds = torch.stack(preds, dim=0)
            probs = torch.nn.functional.softmax(preds, dim=2).cpu().numpy()
            scores = probs[SMILE_ATTR_IDX_NEW] @ self.smile_classes

            outputs.extend(probs[SMILE_ATTR_IDX_NEW].tolist() if return_prob else scores.tolist())

        # Return single result if only one input
        return outputs[0] if images.shape[0] == 1 else np.array(outputs)


    def __call__(self, *args, **kwds):
        """Classify images using the model."""
        # If called, run classification
        return self.classify(*args, **kwds)