# Script to run the LSO algorithm on the FFHQ dataset

# Device and seed
device="cuda"
seed=42

# Dataloader
img_dir="/pfs/work7/workspace/scratch/ma_mgraevin-optdif/data/ffhq/images1024x1024"
img_tensor_dir="/pfs/work7/workspace/scratch/ma_mgraevin-optdif/data/ffhq/pt_images"
attr_path="/pfs/work7/workspace/scratch/ma_mgraevin-optdif/data/ffhq/ffhq_smile_scores.json"
max_property_value=5
min_property_value=4 #0
batch_size=128
num_workers=4

# Weighter
weight_type="uniform"

# Weighted Retraining
query_budget=500
retraining_frequency=5
n_retrain_epochs=0.1
n_init_retrain_epochs=1
result_path="/pfs/work7/workspace/scratch/ma_mgraevin-optdif/results/debug_00/"
sd_vae_path="stabilityai/stable-diffusion-3.5-medium"
latent_vae_config_path="/pfs/work7/workspace/scratch/ma_mgraevin-optdif/models/latent_vae/configs/sd35m_to_128d.json"
predictor_attr_file="/home/ma/ma_ma/ma_mgraevin/pfs5wor7/ma_mgraevin-optdif/models/classifier/celeba_smile/attributes.json"
predictor_path="/pfs/work7/workspace/scratch/ma_mgraevin-optdif/models/classifier/celeba_smile/predictor_128_scaled3.pth.tar"
scaled_predictor=True

# DNGO
n_out=5
n_starts=2 #20
n_samples=10 #10000
n_rand_points=60  #8000
n_best_points=20 #2000
sample_distribution="normal"
opt_method="SLSQP"
opt_constraint_threshold=-94
opt_constraint_strategy="gmm_fit"
n_gmm_components=3 #10
sparse_out=True

# Initialize Conda for the current shell
eval "$(conda shell.bash hook)"

# Activate the conda environment
conda activate optdif1

# Run the Python script with specified arguments
python /home/ma/ma_ma/ma_mgraevin/pfs5wor7/ma_mgraevin-optdif/src/lso.py \
    --device $device \
    --seed $seed \
    --img_dir $img_dir \
    --img_tensor_dir $img_tensor_dir \
    --attr_path $attr_path \
    --max_property_value $max_property_value \
    --min_property_value $min_property_value \
    --batch_size $batch_size \
    --num_workers $num_workers \
    --weight_type $weight_type \
    --rank_weight_k $rank_weight_k \
    --query_budget $query_budget \
    --retraining_frequency $retraining_frequency \
    --n_retrain_epochs $n_retrain_epochs \
    --n_init_retrain_epochs $n_init_retrain_epochs \
    --result_path $result_path \
    --sd_vae_path $sd_vae_path \
    --latent_vae_config_path $latent_vae_config_path \
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