#!/bin/bash
#SBATCH --job-name=attack_greedy_wo             # Job name
#SBATCH --output=attack_greedy_wo_output.log    # Standard output and error log
#SBATCH --error=attack_greedy_wo_error.log      # Error log
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
conda activate jailbreak
    
python attack.py \
    --model Llama-2-7b-chat-hf-mod-b \
    --use_greedy
