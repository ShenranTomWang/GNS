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

RF=1
RECONNECTION_FREQUENCY=1

python train_or_infer.py \
    --reconnection_frequency ${RECONNECTION_FREQUENCY} \
    infer \
        --model_path /ubc/cs/home/s/shenranw/scratch/GNS/train_logs/rf=${RF}/run0/model.pth \
        --logdir /ubc/cs/home/s/shenranw/scratch/GNS/rf=${RF}/rollouts

python render_rollout.py \
    --rollout_path \
    /ubc/cs/home/s/shenranw/scratch/GNS/rf=${RF}/rollouts/rollout_0.pkl