#!/bin/bash

#SBATCH --job-name=train_latent_vae          # Job name
#SBATCH --output=logs/latent_vae/v20_%j.out  # Output log file
#SBATCH --error=logs/latent_vae/v20_%j.err   # Error log file
#SBATCH --time=3-00:00:00                    # Maximum runtime (hh:mm:ss)
#SBATCH --partition=gpu20                    # Partition to submit the job to
#SBATCH --gres=gpu:4                         # Request GPU resources
#SBATCH --mem=0                              # Use all available memory

# Dataloader
img_dir="/BS/robust-architectures/work/OptDif/data/ffhq/images1024x1024"
img_tensor_dir="/BS/robust-architectures/work/OptDif/data/ffhq/pt_images"
attr_path="/BS/robust-architectures/work/OptDif/data/ffhq/ffhq_smile_scores.json"
max_property_value=5
min_property_value=0
batch_size=32
num_workers=8
val_split=0.1
data_device="cuda"

# Weighter
weight_type="uniform"

# Model & Training
model_type="LatentVAE"
model_version=22
model_config_path="models/latent_vae/configs/sd35m_to_8k_lpips_disc.yaml"
model_output_dir="models/latent_vae/"
max_epochs=200
device="cuda"
num_devices=4

# Initialize Conda for the current shell
eval "$(conda shell.bash hook)"

# Activate the conda environment
conda activate optdif1

# Run the Python script with specified arguments
python src/run/train_latent_model_ffhq.py \
    --img_dir $img_dir \
    --img_tensor_dir $img_tensor_dir \
    --attr_path $attr_path \
    --max_property_value $max_property_value \
    --min_property_value $min_property_value \
    --batch_size $batch_size \
    --num_workers $num_workers \
    --val_split $val_split \
    --data_device $data_device \
    --aug \
    --weight_type $weight_type \
    --model_type $model_type \
    --model_version $model_version \
    --model_config_path $model_config_path \
    --model_output_dir $model_output_dir \
    --max_epochs $max_epochs \
    --device $device \
    --num_devices $num_devices \
    "$@"