#!/bin/bash

#SBATCH --job-name=train_latent_prior_v7          # Job name
#SBATCH --output=logs/latent_prior/v7_%j.out      # Std‑out log
#SBATCH --error=logs/latent_prior/v7_%j.err       # Std‑err log
#SBATCH --time=2-00:00:00                         # Max runtime (d‑hh:mm:ss)
#SBATCH --partition=gpu20                         # SLURM partition
#SBATCH --gres=gpu:4                              # 4 GPUs
#SBATCH --mem=0                                   # Use all available memory

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

# Latent‑VQ‑VAE‑2
latent_model_config_path="models/latent_vqvae2/version_1_2/hparams.yaml"
latent_model_ckpt_path="models/latent_vqvae2/version_1_2/checkpoints/last.ckpt"

# Prior setup (shared hyperparameters)
prior_type="transformer" # "transformer" or "pixelsnail"
prior_out_dir="models/latent_prior/"
prior_version=7
n_heads=4 #12
lr=2e-4
weight_decay=5e-3
max_epochs=100
device="cuda"
num_devices=4

# Transformer specific parameters
d_model=768
n_layers=12

# PixelSnail specific parameters
n_chan=128
n_blocks=8
dropout=0.1

# Initialize Conda for the current shell
eval "$(conda shell.bash hook)"

# Activate the conda environment
conda activate optdif1

python src/run/train_latent_prior.py \
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
    --latent_model_config_path $latent_model_config_path \
    --latent_model_ckpt_path $latent_model_ckpt_path \
    --prior_type $prior_type \
    --prior_out_dir $prior_out_dir \
    --prior_version $prior_version \
    --n_heads $n_heads \
    --lr $lr \
    --weight_decay $weight_decay \
    --max_epochs $max_epochs \
    --device $device \
    --num_devices $num_devices \
    --d_model $d_model \
    --n_layers $n_layers \
    --n_chan $n_chan \
    --n_blocks $n_blocks \
    --dropout $dropout \
    "$@"