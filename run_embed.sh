#!/bin/bash
#SBATCH --job-name=gpt_neox_embed
#SBATCH --ntasks=128
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:2
#SBATCH --output=slurm_logs/%j.log

module load anaconda3/2021.11 cudatoolkit/11.3

# Activate our conda environment, if the environment is already activated this fails so
# proceed anyway if this is the case.
conda activate gpt-neox || true

python deepy.py embed.py configs/20B_della.yml configs/embed.yml
