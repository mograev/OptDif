#!/bin/bash

#SBATCH --job-name=train_latent_ae         # Job name
#SBATCH --output=logs/latent_ae/v1_%j.out  # Output log file
#SBATCH --error=logs/latent_ae/v1_%j.err   # Error log file
#SBATCH --time=2-00:00:00                  # Maximum runtime (hh:mm:ss)
#SBATCH --partition=gpu20                  # Partition to submit the job to
#SBATCH --gres=gpu:4                       # Request GPU resources

# Dataloader
img_dir="data/ffhq/images1024x1024"
attr_path="data/ffhq/smile_scores.json"
max_property_value=5
min_property_value=0
batch_size=32
num_workers=8
val_split=0.1

# Model & Training
model_type="LatentAE"
model_version=1
model_config_path="models/latent_ae/configs/sd35m_to_512d_lpips_disc.yaml"
model_output_dir="models/latent_ae/"
max_epochs=100

# Initialize Conda for the current shell
eval "$(conda shell.bash hook)"

# Activate the conda environment
conda activate optdif1

# Run the Python script with specified arguments (using srun for SLURM)
python src/run/train_latent_model_ffhq.py \
    --img_dir $img_dir \
    --attr_path $attr_path \
    --max_property_value $max_property_value \
    --min_property_value $min_property_value \
    --batch_size $batch_size \
    --num_workers $num_workers \
    --val_split $val_split \
    --model_type $model_type \
    --model_version $model_version \
    --model_config_path $model_config_path \
    --model_output_dir $model_output_dir \
    --max_epochs $max_epochs \
    "$@"