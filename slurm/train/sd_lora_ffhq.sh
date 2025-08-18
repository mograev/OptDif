#!/bin/bash

#SBATCH --job-name=train_sd_lora          # Job name
#SBATCH --output=logs/sd_lora/v8_%j.out   # Output log file
#SBATCH --error=logs/sd_lora/v8_%j.err    # Error log file
#SBATCH --time=12:00:00                   # Maximum runtime (hh:mm:ss)
#SBATCH --partition=gpu24                 # Partition to submit the job to
#SBATCH --gres=gpu:8                      # Request GPU resources
#SBATCH --mem=512G                        # Request memory

# Dataloader
img_dir="data/ffhq/images1024x1024"
attr_path="data/ffhq/smile_scores.json"
max_property_value=2
min_property_value=0
batch_size=16
num_workers=8
val_split=0.1

# Model & Training
model_version=8
model_output_dir="models/sd_lora/version_$model_version"
struct_adapter="hed" # depth, hed, none
max_epochs=100
device="cuda"
num_devices=8

# Initialize Conda for the current shell
eval "$(conda shell.bash hook)"

# Activate the conda environment
conda activate optdif1

# Run the Python script with specified arguments, using accelerate
accelerate launch \
    --num_processes $num_devices \
    --mixed_precision "bf16" \
    src/run/train_sd_lora_ffhq.py \
    --img_dir $img_dir \
    --attr_path $attr_path \
    --max_property_value $max_property_value \
    --min_property_value $min_property_value \
    --batch_size $batch_size \
    --num_workers $num_workers \
    --val_split $val_split \
    --model_version $model_version \
    --model_output_dir $model_output_dir \
    --struct_adapter $struct_adapter \
    --max_epochs $max_epochs \
    --device $device \
    --num_devices $num_devices \
    "$@"