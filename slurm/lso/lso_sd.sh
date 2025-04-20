#!/bin/bash

#SBATCH --job-name=lso_sd35m              # Job name
#SBATCH --output=logs/lso_sd35m_%j.out    # Output log file
#SBATCH --error=logs/lso_sd35m_%j.err     # Error log file
#SBATCH --time=30                         # Maximum runtime (hh:mm:ss)
#SBATCH --partition=dev_gpu_h100          # Partition to submit the job to
#SBATCH --gres=gpu                        # Request GPU resources

# Device and seed
device="cuda"
seed=42

# Dataloader
img_dir="data/ffhq/images1024x1024"
img_tensor_dir="data/ffhq/pt_images"
attr_path="data/ffhq/ffhq_smile_scores.json"
max_property_value=2 #5
min_property_value=1 #0
batch_size=32 #128
num_workers=4 #4
val_split=0.
data_device="cuda"

# Weighter
weight_type="uniform"

# Weighted Retraining
query_budget=500
retraining_frequency=5
n_retrain_epochs=0.1
n_init_retrain_epochs=1
result_path="results/debug_01/"
sd_vae_path="stabilityai/stable-diffusion-3.5-medium"
predictor_attr_file="models/classifier/celeba_smile/attributes.json"
predictor_path="models/classifier/celeba_smile/predictor_128_scaled3.pth.tar"
scaled_predictor=True

# DNGO
n_out=5
n_starts=20 #20
n_samples=10000 #10000
n_rand_points=8000  #8000
n_best_points=2000 #2000
sample_distribution="normal"
opt_method="SLSQP"
opt_constraint_threshold=-94
opt_constraint_strategy="gmm_fit"
n_gmm_components=10 #10
sparse_out=True

# Initialize Conda for the current shell
eval "$(conda shell.bash hook)"

# Activate the conda environment
conda activate optdif1

# Run the Python script with specified arguments
srun python src/lso_sd.py \
    --device $device \
    --seed $seed \
    --img_dir $img_dir \
    --img_tensor_dir $img_tensor_dir \
    --attr_path $attr_path \
    --max_property_value $max_property_value \
    --min_property_value $min_property_value \
    --batch_size $batch_size \
    --num_workers $num_workers \
    --val_split $val_split \
    --data_device $data_device \
    --weight_type $weight_type \
    --query_budget $query_budget \
    --retraining_frequency $retraining_frequency \
    --n_retrain_epochs $n_retrain_epochs \
    --n_init_retrain_epochs $n_init_retrain_epochs \
    --result_path $result_path \
    --sd_vae_path $sd_vae_path \
    --predictor_path $predictor_path \
    --scaled_predictor $scaled_predictor \
    --predictor_attr_file $predictor_attr_file \
    --n_out $n_out \
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
    "$@"