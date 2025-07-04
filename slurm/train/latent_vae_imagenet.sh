#!/bin/bash

#SBATCH --job-name=train_latent_vae          # Job name
#SBATCH --output=logs/latent_vae/v21_%j.out  # Output log file
#SBATCH --error=logs/latent_vae/v21_%j.err   # Error log file
#SBATCH --time=7-00:00:00                    # Maximum runtime (hh:mm:ss)
#SBATCH --partition=gpu20                    # Partition to submit the job to
#SBATCH --gres=gpu:4                         # Request GPU resources

# Dataloader
img_dir="/BS/spectral-gan2/nobackup/ILSVRC2012"
batch_size=16
num_workers=8
data_device="cuda"

# Model & Training
model_type="LatentVAE"
model_version=21
model_config_path="models/latent_vae/configs/sd35m_to_512d_lpips_disc.yaml"
model_output_dir="models/latent_vae/"
max_epochs=200
device="cuda"
num_devices=4

# Initialize Conda for the current shell
eval "$(conda shell.bash hook)"

# Activate the conda environment
conda activate optdif1

# Run the Python script with specified arguments
python src/run/train_latent_model_imagenet.py \
    --img_dir $img_dir \
    --batch_size $batch_size \
    --num_workers $num_workers \
    --data_device $data_device \
    --aug $aug \
    --model_type $model_type \
    --model_version $model_version \
    --model_config_path $model_config_path \
    --model_output_dir $model_output_dir \
    --max_epochs $max_epochs \
    --device $device \
    --num_devices $num_devices \
    "$@"