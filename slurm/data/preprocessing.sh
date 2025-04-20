#!/bin/bash

#SBATCH --job-name=preprocessing                                    # Job name
#SBATCH --chdir=/pfs/work9/workspace/scratch/ma_mgraevin-optdif     # Working directory
#SBATCH --output=logs/preprocessing_%j.out                          # Output log file
#SBATCH --error=logs/preprocessing_%j.err                           # Error log file
#SBATCH --time=03:00:00                                             # Maximum runtime (hh:mm:ss)
#SBATCH --partition=cpu                                             # Partition to submit the job to
#SBATCH --ntasks=1                                                  # Number of tasks (processes)
#SBATCH --cpus-per-task=4                                           # Number of CPU cores per task
#SBATCH --mem=16G                                                   # Memory per node

# Initialize conda
eval "$(conda shell.bash hook)"

# Activate conda environment
conda activate optdif1

# Run the Python script
srun python src/run/ffhq_preprocessing.py