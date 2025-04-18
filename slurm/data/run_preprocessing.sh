#!/bin/bash

#SBATCH --job-name=preprocessing            # Job name
#SBATCH --output=logs/preprocessing_%j.out  # Output log file
#SBATCH --error=logs/preprocessing_%j.err   # Error log file
#SBATCH --time=03:00:00                     # Maximum runtime (hh:mm:ss)
#SBATCH --partition=single                  # Partition to submit the job to
#SBATCH --ntasks=1                          # Number of tasks (processes)
#SBATCH --cpus-per-task=4                   # Number of CPU cores per task
#SBATCH --mem=16G                           # Memory per node

# Initialize conda
eval "$(conda shell.bash hook)"

# Activate conda environment
conda activate /pfs/work7/workspace/scratch/ma_mgraevin-optdif/.conda/envs/optdif1

# Run the Python script
python /home/ma/ma_ma/ma_mgraevin/pfs5wor7/ma_mgraevin-optdif/src/dataloader/run_preprocessing.py