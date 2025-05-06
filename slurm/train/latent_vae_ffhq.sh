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

# Clear interfering Python paths (when using JupyterHub)
#unset PYTHONPATH
#export PYTHONPATH=/pfs/work9/workspace/scratch/ma_mgraevin-optdif:$PYTHONPATH

# Initialize Conda for the current shell
eval "$(conda shell.bash hook)"

# Activate the conda environment
conda activate optdif1

# Run the Python script with specified arguments (using srun for SLURM)
PYTORCH_DISTRIBUTED_DEBUG=DETAIL python src/run/train_latent_vae_ffhq.py \
    --img_dir $img_dir \
    --img_tensor_dir $img_tensor_dir \
    --attr_path $attr_path \
    --max_property_value $max_property_value \
    --min_property_value $min_property_value \
    --batch_size $batch_size \
    --num_workers $num_workers \
    --val_split $val_split \
    --weight_type $weight_type \
    --data_device $data_device \
    --aug
    "$@"