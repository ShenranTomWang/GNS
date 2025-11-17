#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=4
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --account=st-hgonen-1-gpu
#SBATCH --output=train.log
#SBATCH --error=train.log
#SBATCH --mail-user=shenranw@student.ubc.ca
#SBATCH --mail-type=ALL
#SBATCH --job-name=train

source /home/shenranw/.bashrc
cd /ubc/cs/home/s/shenranw/GNS
source ../scratch/envs/GNS/.venv/bin/activate

bash ./download_dataset.sh WaterDropSample /ubc/cs/home/s/shenranw/scratch/datasets/

python train_or_infer.py \
    --data_dir /ubc/cs/home/s/shenranw/scratch/datasets/WaterDropSample \
    train \
        --logdir /ubc/cs/home/s/shenranw/scratch/GNS/train_logs