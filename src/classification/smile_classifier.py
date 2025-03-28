import torch
import numpy as np

import json
import logging

from src.classification.resnet50 import resnet50
from src.classification.temperature_scaling import ModelWithTemperature

SMILE_ATTR_IDX_NEW = 3
SMILE_ATTR_IDX_OLD = 32

INPUT_SIZE = 224 # ResNet50 input size


def _setup_logger(logfile, log_level):
    """
    Set up a logger to log messages to a file and the console.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    # Create a file handler to log to a file
    file_handler = logging.FileHandler(logfile)
    file_handler.setLevel(log_level)

    # Create a console handler to log to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    # Create a formatter and set it for both handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

class SmileClassifier:
    def __init__(self, model_path, attr_file, scaled_model_path=None, device="cpu", logfile="smile_classifier.log", log_level=logging.INFO):
        """
        Initialize the SmileClassifier.
        Args:
            model_path (str): Path to the model checkpoint.
            attr_file (str): Path to the attribute file.
            scaled_model_path (str): Path to the temperature-scaled model checkpoint.
            device (str): Device to use for computation ("cpu" or "cuda").
            logfile (str): Path to the log file.
            verbose (bool): If True, enable verbose logging.
        """
        # Set up logger
        self.logger = _setup_logger(logfile, log_level)

        # Model initialization
        self.logger.info("Initializing SmileClassifier")
        self.model = resnet50(attr_file=attr_file)

        self.device = device
        self.logger.debug(f"Using device: {device}")

        # Load model checkpoint
        self.logger.debug(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=torch.device(self.device))
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model.eval()  # Set to evaluation mode

        # Load temperature scaling
        if scaled_model_path:
            self.logger.debug(f"Loading scaled model from {scaled_model_path}")
            self.checkpoint_scaled_model = torch.load(scaled_model_path, map_location=torch.device(device))
            self.model=ModelWithTemperature(self.model)
            self.model.load_state_dict(self.checkpoint_scaled_model, strict=True)
            self.model.eval()

        # Load attribute mapping from JSON file
        self.logger.debug(f"Loading attribute mapping from {attr_file}")
        with open(attr_file, 'r') as f:
            self.attr_data = json.load(f)

        # Extract smile classes
        self.smile_classes = self.attr_data["attr_info"][f"{SMILE_ATTR_IDX_OLD}"]["value"]
        self.smile_classes = np.array(self.smile_classes)


    def classify(self, tensor_paths, batch_size=128, return_prob=False):
        """
        Classify a single image or a batch of images
        """

        # Ensure image_paths is a list
        if isinstance(tensor_paths, str):
            tensor_paths = [tensor_paths]
            self.logger.debug(f"Single image path provided: {tensor_paths[0]}")

        # Process images in batches if necessary
        all_outputs = []
        for i in range(0, len(tensor_paths), batch_size):
            # Get the current batch
            batch = tensor_paths[i : i + batch_size]
            self.logger.info(f"Processing batch {i // batch_size + 1} of size {len(batch)}")

            # Load and preprocess image tensors
            input_tensors = []
            for tensor_path in batch:
                # Load image tensor
                img = torch.load(tensor_path)
                # Add to list
                input_tensors.append(img)
            # Stack tensors into a single tensor
            input_tensors = torch.stack(input_tensors, dim=0)

            # Ensure input tensors are in the correct format
            if input_tensors.dim() == 3:
                input_tensors = input_tensors.unsqueeze(0)
            elif input_tensors.dim() != 4:
                error_msg = f"Input tensor must be 3D or 4D, got {input_tensors.dim()}D"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            # Ensure input tensors have the correct image size
            input_tensors = torch.nn.functional.interpolate(input_tensors, size=(224, 224), mode="bilinear", align_corners=False, antialias=True)

            # Move tensors to the appropriate device
            input_tensors = input_tensors.to(self.device)

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

            if return_prob:
                # Store probabilities and smile scores
                all_outputs.extend(probabilities[SMILE_ATTR_IDX_NEW].tolist())
            else:
                # Store only smile scores
                all_outputs.extend(smile_scores.tolist())

        # Convert to numpy array
        all_outputs = np.array(all_outputs)
        self.logger.info(f"Output shape: {all_outputs.shape}")

        # Return a single value for single input or an array otherwise
        return all_outputs[0] if len(tensor_paths) == 1 else all_outputs


    def __call__(self, *args, **kwds):
        # If called, run classification
        return self.classify(*args, **kwds)