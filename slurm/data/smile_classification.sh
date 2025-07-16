#!/bin/bash

#SBATCH --job-name=smile_classification            # Job name
#SBATCH --output=logs/smile_classification/%j.out  # Output log file
#SBATCH --error=logs/smile_classification/%j.err   # Error log file
#SBATCH --time=06:00:00                            # Maximum runtime (hh:mm:ss)
#SBATCH --partition=gpu20                          # Partition to submit the job to
#SBATCH --gres=gpu:1                               # Request GPU resources

# Initialize conda
eval "$(conda shell.bash hook)"

# Activate conda environment
conda activate optdif1

# Run the Python script
srun python src/run/smile_classification.py