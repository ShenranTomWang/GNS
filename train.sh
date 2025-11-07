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
conda activate GNS

export HF_HOME=/home/shenranw/scratch/tmp/transformers_cache
export TRITON_CACHE_DIR=/home/shenranw/scratch/tmp/triton_cache

cd /home/shenranw/GNS
python train_or_infer.py