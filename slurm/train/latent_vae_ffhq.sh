# Script to run LatentVAE training on the FFHQ dataset

# Dataloader
img_dir="/pfs/work7/workspace/scratch/ma_mgraevin-optdif/data/ffhq/images1024x1024"
img_tensor_dir="/pfs/work7/workspace/scratch/ma_mgraevin-optdif/data/ffhq/pt_images"
attr_path="/pfs/work7/workspace/scratch/ma_mgraevin-optdif/data/ffhq/ffhq_smile_scores.json"
max_property_value=5
min_property_value=0
batch_size=128
num_workers=4
val_split=0.1

# Weighter
weight_type="uniform"

# Initialize Conda for the current shell
eval "$(conda shell.bash hook)"

# Activate the conda environment
conda activate optdif1

# Run the Python script with specified arguments
python /home/ma/ma_ma/ma_mgraevin/pfs5wor7/ma_mgraevin-optdif/src/run/train_latent_vae_ffhq.py \
    --img_dir $img_dir \
    --img_tensor_dir $img_tensor_dir \
    --attr_path $attr_path \
    --max_property_value $max_property_value \
    --min_property_value $min_property_value \
    --batch_size $batch_size \
    --num_workers $num_workers \
    --val_split $val_split \
    --weight_type $weight_type \
    "$@"