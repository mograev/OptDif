#!/bin/bash

#SBATCH --job-name=train_latent_autoencoder                 # Job name
#SBATCH --output=logs/train_latent_autoencoder_512d_%j.out  # Output log file
#SBATCH --error=logs/train_latent_autoencoder_512d_%j.err   # Error log file
#SBATCH --time=12:00:00                                     # Maximum runtime (hh:mm:ss)
#SBATCH --partition=gpu_4_h100                              # Partition to submit the job to
#SBATCH --gres=gpu                                          # Request GPU resources

# Dataloader
img_dir="data/ffhq/images1024x1024"
img_tensor_dir="data/ffhq/pt_images"
attr_path="data/ffhq/ffhq_smile_scores.json"
max_property_value=5
min_property_value=0
batch_size=128
num_workers=8
val_split=0.1
device="cuda"

# Weighter
weight_type="uniform"

# Initialize Conda for the current shell
eval "$(conda shell.bash hook)"

# Activate the conda environment
conda activate optdif1

# Run the Python script with specified arguments (using srun for SLURM)
srun python src/run/train_latent_autoencoder_ffhq.py \
    --img_dir $img_dir \
    --img_tensor_dir $img_tensor_dir \
    --attr_path $attr_path \
    --max_property_value $max_property_value \
    --min_property_value $min_property_value \
    --batch_size $batch_size \
    --num_workers $num_workers \
    --val_split $val_split \
    --weight_type $weight_type \
    --device $device \
    --aug
    "$@"