#!/bin/bash
#SBATCH --job-name=evaluate_gpt                 # Job name
#SBATCH --output=evaluate_gpt_output.log        # Standard output and error log
#SBATCH --error=evaluate_gpt_error.log          # Error log
#SBATCH --nodes=1                               # Run on a single node
#SBATCH --ntasks=1                              # Total number of tasks
#SBATCH --cpus-per-task=8                       # CPU cores per task
#SBATCH --mem=32G                               # Total memory per node
#SBATCH --time=4:00:00                          # Time limit hrs:min:sec
#SBATCH --gres=gpu:1                            # Request 1 GPU
#SBATCH --mail-type=END,FAIL                    # Send email on job completion or failure
#SBATCH --mail-user=sa3937@princeton.edu

module load anaconda3/2024.6
conda init
conda activate jailbreak

python evaluate_gpt.py \
    --model gpt-4.1 \
    --config 'exploited' \
    --evaluator_path /scratch/network/sa3937/cos484/Jailbreak_LLM-main/evaluator \
    --scorer_path /scratch/network/sa3937/cos484/Jailbreak_LLM-main/scorer