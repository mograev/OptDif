#!/bin/bash

#SBATCH --job-name=lso_sd_gp               # Job name
#SBATCH --output=logs/lso/sd_gp_01_%j.out  # Output log file
#SBATCH --error=logs/lso/sd_gp_01_%j.err   # Error log file
#SBATCH --time=8:00:00                     # Maximum runtime (hh:mm:ss)
#SBATCH --partition=gpu20                  # Partition to submit the job to
#SBATCH --gres=gpu:1                       # Request GPU resources
#SBATCH --exclude=gpu20-45

# Device and seed
device="cuda"
seed=42

# Dataloader
img_dir="data/ffhq/images1024x1024"
attr_path="data/ffhq/smile_scores.json"
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
result_path="results/sd_gp_01/"
sd_vae_path="stabilityai/stable-diffusion-3.5-medium" #"models/sd_vae/version_0/huggingface"
predictor_attr_file="models/classifier/celeba_smile/attributes.json"
predictor_path="models/classifier/celeba_smile/predictor_128.pth.tar"

# Optimization
opt_strategy="GP"
n_starts=20 #20
n_samples=1000 # 10000
n_rand_points=8000  #8000
n_best_points=2000 #2000
sample_distribution="train_data" # "uniform", "normal", or "train_data"
opt_method="SLSQP"
opt_constraint="GMM"
n_gmm_components=10
sparse_out=True

# Feature Selection
feature_selection="None" # "PCA", "FI", or None
feature_selection_dims=512 # 512
feature_selection_model_path="models/feature_selection/sd_latents_pca_model.pkl"

# Initialize Conda for the current shell
eval "$(conda shell.bash hook)"

# Activate the conda environment
conda activate optdif1

# Run the Python script with specified arguments
CUDA_VISIBLE_DEVICES=0 python src/lso_sd.py \
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
    --predictor_attr_file $predictor_attr_file \
    --predictor_path $predictor_path \
    --opt_strategy $opt_strategy \
    --n_starts $n_starts \
    --n_samples $n_samples \
    --n_rand_points $n_rand_points \
    --n_best_points $n_best_points \
    --sample_distribution $sample_distribution \
    --opt_method $opt_method \
    --opt_constraint $opt_constraint \
    --n_gmm_components $n_gmm_components \
    --sparse_out $sparse_out \
    --feature_selection $feature_selection \
    --feature_selection_dims $feature_selection_dims \
    --feature_selection_model_path $feature_selection_model_path \
    "$@"