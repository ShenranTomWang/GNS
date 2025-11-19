#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=12:0:0
#SBATCH --partition=nlpgpo

source /home/shenranw/.bashrc
cd /ubc/cs/home/s/shenranw/GNS
source ../scratch/envs/GNS/.venv/bin/activate

# bash ./download_dataset.sh WaterDropSample /ubc/cs/home/s/shenranw/scratch/datasets/

python train_or_infer.py \
    infer \
        --model_path /ubc/cs/home/s/shenranw/scratch/GNS/train_logs/run6/model.pth \
        --logdir /ubc/cs/home/s/shenranw/scratch/GNS/rollouts