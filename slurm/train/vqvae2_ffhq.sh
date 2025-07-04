#!/bin/bash

#SBATCH --job-name=train_vqvae2           # Job name
#SBATCH --output=logs/vqvae2/v0_%j.out    # Output log file
#SBATCH --error=logs/vqvae2/v0_%j.err     # Error log file
#SBATCH --time=1-00:00:00                 # Maximum runtime (d-hh:mm:ss)
#SBATCH --partition=gpu20                 # Partition to submit the job to
#SBATCH --gres=gpu:4                      # Request GPU resources

# Dataloader
img_dir="data/ffhq/images1024x1024"
attr_path="data/ffhq/ffhq_smile_scores.json"
max_property_value=5
min_property_value=0
batch_size=16
num_workers=8
val_split=0.1

# Model & Training
model_version="0_2"
model_config_path="models/vqvae2/configs/img_to_2k_lpips_disc.yaml"
model_output_dir="models/vqvae2/"
max_epochs=200
device="cuda"
num_devices=4

# Initialize Conda for the current shell
eval "$(conda shell.bash hook)"

# Activate the conda environment
conda activate optdif1

# Run the Python script with specified arguments
python src/run/train_vqvae2_ffhq.py \
    --img_dir $img_dir \
    --attr_path $attr_path \
    --max_property_value $max_property_value \
    --min_property_value $min_property_value \
    --batch_size $batch_size \
    --num_workers $num_workers \
    --val_split $val_split \
    --model_version $model_version \
    --model_config_path $model_config_path \
    --model_output_dir $model_output_dir \
    --max_epochs $max_epochs \
    --device $device \
    --num_devices $num_devices \
    "$@"