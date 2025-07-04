#!/bin/bash

#SBATCH --job-name=lso_latent_vqvae_dngo               # Job name
#SBATCH --output=logs/lso/latent_vqvae_dngo_04_%j.out  # Output log file
#SBATCH --error=logs/lso/latent_vqvae_dngo_04_%j.err   # Error log file
#SBATCH --time=4:00:00                                 # Maximum runtime (hh:mm:ss)
#SBATCH --partition=gpu20                              # Partition to submit the job to
#SBATCH --gres=gpu:1                                   # Request GPU resources

# Device and seed
device="cuda"
seed=42

# Dataloader
img_dir="data/ffhq/images1024x1024"
attr_path="data/ffhq/ffhq_smile_scores.json"
max_property_value=2
min_property_value=0
batch_size=16
num_workers=8
val_split=0

# Weighter
weight_type="uniform"

# Weighted Retraining
query_budget=50 #500
retraining_frequency=5
n_retrain_epochs=0 #0.1
n_init_retrain_epochs=0 #1
result_path="results/latent_vqvae_dngo_05/"
sd_vae_path="stabilityai/stable-diffusion-3.5-medium"
latent_model_config_path="models/latent_vqvae2/version_0/hparams.yaml"
latent_model_ckpt_path="models/latent_vqvae2/version_0/checkpoints/last.ckpt"
predictor_attr_file="models/classifier/celeba_smile/attributes.json"
predictor_path="models/classifier/celeba_smile/predictor_128_scaled3.pth.tar"
scaled_predictor=True

# Optimization
opt_strategy="DNGO"
n_starts=20 #20
n_samples=10000 # 10000
n_rand_points=800  #8000
n_best_points=200 #2000
sample_distribution="train_data" # "uniform", "normal", or "train_data"
opt_method="SLSQP"
opt_constraint_threshold=-15000000 #100000000
opt_constraint_strategy="gmm_fit"
n_gmm_components=10
sparse_out=True

# Feature Selection
feature_selection="FI" # "FI", "PCA", "None"
feature_selection_dims=512
feature_selection_model_path="models/feature_selection/latents_fi_model.pkl"

# Initialize Conda for the current shell
eval "$(conda shell.bash hook)"

# Activate the conda environment
conda activate optdif1

# Run the Python script with specified arguments
CUDA_VISIBLE_DEVICES=1 python src/lso_latent_vqvae.py \
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
    --sd_vae_path $sd_vae_path \
    --latent_model_config_path $latent_model_config_path \
    --latent_model_ckpt_path $latent_model_ckpt_path \
    --predictor_attr_file $predictor_attr_file \
    --predictor_path $predictor_path \
    --scaled_predictor $scaled_predictor \
    --opt_strategy $opt_strategy \
    --n_starts $n_starts \
    --n_samples $n_samples \
    --n_rand_points $n_rand_points \
    --n_best_points $n_best_points \
    --sample_distribution $sample_distribution \
    --opt_method $opt_method \
    --opt_constraint_threshold $opt_constraint_threshold \
    --opt_constraint_strategy $opt_constraint_strategy \
    --n_gmm_components $n_gmm_components \
    --sparse_out $sparse_out \
    --feature_selection $feature_selection \
    --feature_selection_dims $feature_selection_dims \
    --feature_selection_model_path $feature_selection_model_path \
    "$@"