#!/bin/bash
#SBATCH --job-name=download                     # Job name
#SBATCH --output=download_output.log            # Standard output and error log
#SBATCH --error=download_error.log              # Error log
#SBATCH --nodes=1                               # Run on a single node
#SBATCH --ntasks=1                              # Total number of tasks
#SBATCH --cpus-per-task=8                       # CPU cores per task
#SBATCH --mem=64G                               # Total memory per node
#SBATCH --time=24:00:00                         # Time limit hrs:min:sec
#SBATCH --gres=gpu:1                            # Request 1 GPU
#SBATCH --mail-type=END,FAIL                    # Send email on job completion or failure
#SBATCH --mail-user=sa3937@princeton.edu

module load anaconda3/2024.6
conda init
conda activate misalign
    
python download_models2.py