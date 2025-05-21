#!/bin/bash

#SBATCH --job-name=train_latent_vae      # Job name
#SBATCH --output=logs/latent_vae/%j.out  # Output log file
#SBATCH --error=logs/latent_vae/%j.err   # Error log file
#SBATCH --time=2:00:00                   # Maximum runtime (hh:mm:ss)
#SBATCH --partition=gpu_a100_il          # Partition to submit the job to
#SBATCH --gres=gpu:4                     # Request GPU resources

# Dataloader
img_dir="data/ffhq/images1024x1024"
img_tensor_dir="data/ffhq/pt_images"
attr_path="data/ffhq/ffhq_smile_scores.json"
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
model_version=18
model_config_path="models/latent_vae/configs/sd35m_to_8k_lpips_disc.yaml"
model_output_dir="models/latent_vae/"
max_epochs=100
device="cuda"
num_devices=4

# Clear interfering Python paths (when using JupyterHub)
unset PYTHONPATH
export PYTHONPATH=/pfs/work9/workspace/scratch/ma_mgraevin-optdif:$PYTHONPATH

# Initialize Conda for the current shell
eval "$(conda shell.bash hook)"

# Activate the conda environment
conda activate optdif1

# Run the Python script with specified arguments (using srun for SLURM)
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