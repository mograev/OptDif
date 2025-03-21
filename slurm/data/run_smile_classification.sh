#!/bin/bash

#SBATCH --job-name=smile_classification    # Job name
#SBATCH --output=logs/smile_classification_%j.out  # Output log file (%j will be replaced with the job ID)
#SBATCH --error=logs/smile_classification_%j.err   # Error log file (%j will be replaced with the job ID)
#SBATCH --time=06:00:00                   # Maximum runtime (hh:mm:ss)
#SBATCH --partition=single                # Partition to submit the job to
#SBATCH --ntasks=1                        # Number of tasks (processes)
#SBATCH --cpus-per-task=4                 # Number of CPU cores per task
#SBATCH --mem=8G                         # Memory per node

# Initialize conda
eval "$(conda shell.bash hook)"

# Activate conda environment
conda activate /pfs/work7/workspace/scratch/ma_mgraevin-optdif/.conda/envs/optdif1

# Run the Python script
python /home/ma/ma_ma/ma_mgraevin/pfs5wor7/ma_mgraevin-optdif/src/classification/initial_smile_classification.py