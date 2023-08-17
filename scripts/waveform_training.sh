#!/bin/sh
#SBATCH --job-name=ad_test
#SBATCH --gres=gpu:1
#SBATCH --mem=1000
#SBATCH --cpus-per-task=1
#SBATCH --partition=MEDIUM-G1

. /home/ICTDOMAIN/d22127229/torchenv/bin/activate

nvidia-smi

python --version

srun python train_waveform_model_esc.py
