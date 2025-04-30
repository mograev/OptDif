#!/bin/bash

#SBATCH --job-name=smile_classification            # Job name
#SBATCH --output=logs/smile_classification/%j.out  # Output log file
#SBATCH --error=logs/smile_classification/%j.err   # Error log file
#SBATCH --time=06:00:00                            # Maximum runtime (hh:mm:ss)
#SBATCH --partition=cpu                            # Partition to submit the job to
#SBATCH --cpus-per-task=4                          # Number of CPU cores per task
#SBATCH --mem=8G                                   # Memory per node

# Initialize conda
eval "$(conda shell.bash hook)"

# Activate conda environment
conda activate optdif1

# Run the Python script
srun python src/classification/initial_smile_classification.py