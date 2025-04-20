import os
import json

from PIL import Image
import torch
from torchvision import transforms

def preprocess_ffhq(img_dir, img_tensor_dir, attr_path, max_property_value=5, min_property_value=0):
    # Load attribute JSON file
    with open(attr_path, 'r') as f:
        attr_dict = json.load(f)
    
    # Get sorted filenames that meet attribute criteria
    filenames = []
    for key in sorted(attr_dict.keys()):
        filename = key.split('.')[0]
        if min_property_value <= attr_dict[key] <= max_property_value:
            filenames.append(filename)
    
    print("Preprocessing the FFHQ dataset...")

    # Define preprocessing transformation
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    os.makedirs(img_tensor_dir, exist_ok=True)

    # Only process images that haven't been converted yet
    filenames_to_convert = [
        filename for filename in filenames 
        if not os.path.exists(os.path.join(img_tensor_dir, f"{filename}.pt"))
    ]
    
    for filename in filenames_to_convert:
        image_path = os.path.join(img_dir, f"{filename}.png")
        image = Image.open(image_path).convert("RGB")
        tensor = transform(image)
        torch.save(tensor, os.path.join(img_tensor_dir, f"{filename}.pt"))
    
    print("Conversion to .pt format complete. Finished preprocessing the FFHQ dataset.")

if __name__ == "__main__":
    IMG_DIR = "data/ffhq/images1024x1024/"
    IMG_TENSOR_DIR = "data/ffhq/pt_images/"
    ATTR_PATH = "data/ffhq/ffhq_smile_scores.json"

    preprocess_ffhq(IMG_DIR, IMG_TENSOR_DIR, ATTR_PATH)
