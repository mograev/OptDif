from src.dataloader.ffhq import FFHQ

# Configuration
IMG_DIR = "data/ffhq/images1024x1024/"
PT_DIR = "data/ffhq/pt_images/"
ATTR_PATH = "data/ffhq/ffhq_smile_scores.json"

ffhq_dataset = FFHQ(IMG_DIR, PT_DIR, ATTR_PATH, do_preprocess=True)
print(f"Number of images: {ffhq_dataset.num_images}")