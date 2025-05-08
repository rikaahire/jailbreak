#!/bin/bash
#SBATCH --job-name=evaluate_greedy              # Job name
#SBATCH --output=evaluate_greedy_output.log     # Standard output and error log
#SBATCH --error=evaluate_greedy_error.log       # Error log
#SBATCH --nodes=1                               # Run on a single node
#SBATCH --ntasks=1                              # Total number of tasks
#SBATCH --cpus-per-task=8                       # CPU cores per task
#SBATCH --mem=32G                               # Total memory per node
#SBATCH --time=0:10:00                          # Time limit hrs:min:sec
#SBATCH --gres=gpu:1                            # Request 1 GPU
#SBATCH --mail-type=END,FAIL                    # Send email on job completion or failure
#SBATCH --mail-user=sa3937@princeton.edu

module load anaconda3/2024.6
conda init
conda activate jailbreak

python evaluate.py \
    --model Llama-2-7b-chat-hf-mod-b_with_sys_prompt \
    --config 'greedy-only' \
    --evaluator_path /scratch/network/sa3937/cos484/Jailbreak_LLM-main/evaluator \
    --scorer_path /scratch/network/sa3937/cos484/Jailbreak_LLM-main/scorer
