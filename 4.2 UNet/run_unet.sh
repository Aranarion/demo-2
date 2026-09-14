#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch

python3 unet.py
