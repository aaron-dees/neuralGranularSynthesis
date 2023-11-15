#!/bin/sh
#SBATCH --job-name=ad_test
#SBATCH --gres=gpu:1
#SBATCH --mem=1000
#SBATCH --cpus-per-task=1
#SBATCH --partition=LARGE-G2

nvidia-smi

. /home/ICTDOMAIN/d22127229/torchenv/bin/activate

python --version

python train_temporal_model.py
