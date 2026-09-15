#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=8

source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch

python3 GAN.py
