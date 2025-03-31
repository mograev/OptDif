"""
    This is a test metric that checks how much red the image is.
    It is used to validate the optimization process.
    It is not used in the final model.
"""

import torch

def calculate_red_percentage(image):
    """
    Calculate the percentage of red in the given image.

    Args:
        image (torch.Tensor): The input image as a PyTorch tensor with shape (3, height, width).

    Returns:
        float: The percentage of red pixels in the image.
    """

    # Normalize the image tensor to the range [0, 1]
    image = (image - image.min()) / (image.max() - image.min())

    # Extract the red channel
    # Assuming the image tensor is in the format (C, H, W)
    red_channel = image[0, :, :]
    
    # Calculate the total intensity of the red channel
    total_red_intensity = torch.sum(red_channel).item()

    # Calculate the total intensity of all channels
    total_intensity = torch.sum(image).item()

    # Avoid division by zero
    if total_intensity == 0:
        return 0.0

    # Calculate the percentage of red
    red_percentage = total_red_intensity / total_intensity
    return red_percentage