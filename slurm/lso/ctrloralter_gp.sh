#!/bin/bash

#SBATCH --job-name=ctrloralter_gp             # Job name
#SBATCH --output=logs/lso/ctrloralter_gp.out  # Output log file
#SBATCH --error=logs/lso/ctrloralter_gp.err   # Error log file
#SBATCH --time=12:00:00                       # Maximum runtime (hh:mm:ss)
#SBATCH --partition=gpu20                     # Partition to submit the job to
#SBATCH --gres=gpu:1                          # Request GPU resources
#SBATCH --cpus-per-task=16                    # Number of CPU cores per task

# Device and seed
device="cuda"
seed=42 # 42, 43, 44

# Dataloader
img_dir="data/ffhq/images1024x1024"
attr_path="data/ffhq/smile_scores.json"
max_property_value=2
min_property_value=0
batch_size=128
num_workers=8
val_split=0

# Weighter
weight_type="uniform"

# Weighted Retraining
query_budget=100
retraining_frequency=5
n_retrain_epochs=0 # 0.1
n_init_retrain_epochs=0 # 1
result_path="results/ctrloralter_gp/"
predictor_attr_file="models/classifier/celeba_smile/attributes.json"
predictor_path="models/classifier/celeba_smile/predictor_128.pth.tar"

# Stable Diffusion model path
# - SD15: "runwayml/stable-diffusion-v1-5"
# - SDXL: "stabilityai/stable-diffusion-xl-base-1.0"
sd_path="runwayml/stable-diffusion-v1-5"
# Optional Adapter (only for SD15):
struct_adapter="none" # "depth", "hed", "none"
# Checkpoint paths:
# - SD15
#   - Initial:
#       - Style: "src/ctrloralter/checkpoints/sd15-style-cross-160-h"
#       - Depth: "src/ctrloralter/checkpoints/sd15-depth-128-only-res"
#       - HED: "src/ctrloralter/checkpoints/sd15-hed-128-only-res"
#   - Finetuned:
#       - Style: "models/sd_lora/version_9/checkpoints/epoch_053"
#       - Style+Depth: "models/sd_lora/version_7/checkpoints/epoch_053"
#       - Style+HED: "models/sd_lora/version_8/checkpoints/epoch_053"
# - SDXL: "src/ctrloralter/checkpoints/sdxl_b-lora_256"
style_ckpt_path="models/sd_lora/version_9/checkpoints/epoch_053"
struct_ckpt_path="models/sd_lora/version_9/checkpoints/epoch_053"

# Optimization
opt_strategy="GP"
n_rand_points=8000
n_best_points=2000
n_starts=20
n_samples=1000
sample_distribution="train_data" # "train_data", "normal"
opt_method="L-BFGS-B" # "L-BFGS-B", "trust-constr", "SLSQP"
opt_constraint="None" # "None", "GMM"
n_gmm_components=10
sparse_out=True

# Initialize Conda for the current shell
eval "$(conda shell.bash hook)"

# Activate the conda environment
conda activate optdif2

# Run the Python script with specified arguments
CUDA_VISIBLE_DEVICES=0 python src/lso_ctrloralter.py \
    --device $device \
    --seed $seed \
    --img_dir $img_dir \
    --attr_path $attr_path \
    --max_property_value $max_property_value \
    --min_property_value $min_property_value \
    --batch_size $batch_size \
    --num_workers $num_workers \
    --val_split $val_split \
    --weight_type $weight_type \
    --query_budget $query_budget \
    --retraining_frequency $retraining_frequency \
    --n_retrain_epochs $n_retrain_epochs \
    --n_init_retrain_epochs $n_init_retrain_epochs \
    --result_path $result_path \
    --sd_path $sd_path \
    --struct_adapter $struct_adapter \
    --predictor_attr_file $predictor_attr_file \
    --predictor_path $predictor_path \
    --style_ckpt_path $style_ckpt_path \
    --struct_ckpt_path $struct_ckpt_path \
    --opt_strategy $opt_strategy \
    --n_rand_points $n_rand_points \
    --n_best_points $n_best_points \
    --n_starts $n_starts \
    --n_samples $n_samples \
    --sample_distribution $sample_distribution \
    --opt_method $opt_method \
    --opt_constraint $opt_constraint \
    --n_gmm_components $n_gmm_components \
    --sparse_out $sparse_out \
    "$@"