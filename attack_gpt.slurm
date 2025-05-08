#!/bin/bash
#SBATCH --job-name=attack_gpt                   # Job name
#SBATCH --output=attack_gpt_output.log          # Standard output and error log
#SBATCH --error=attack_gpt_error.log            # Error log
#SBATCH --nodes=1                               # Run on a single node
#SBATCH --ntasks=1                              # Total number of tasks
#SBATCH --cpus-per-task=8                       # CPU cores per task
#SBATCH --mem=128G                              # Total memory per node
#SBATCH --time=48:00:00                         # Time limit hrs:min:sec
#SBATCH --gres=gpu:2                            # Request 1 GPU
#SBATCH --mail-type=END,FAIL                    # Send email on job completion or failure
#SBATCH --mail-user=sa3937@princeton.edu

module load anaconda3/2024.6
conda init
conda activate jailbreak
    
python attack_gpt.py \
    --model gpt-3.5-turbo \
    --tune_temp \
    --tune_topp \
    --n_sample 1