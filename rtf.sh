#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=12:0:0
#SBATCH --partition=nlpgpo

source /home/shenranw/.bashrc
cd /ubc/cs/home/s/shenranw/GNS
source ../scratch/envs/GNS/.venv/bin/activate

for RF in 1 2 3
do
    echo "=====================================Using RF=${RF} model======================================="
    for RECONNECTION_FREQUENCY in 1 2 3
    do
        echo "-----------------------Reconnection Frequency: ${RECONNECTION_FREQUENCY}------------------------"
        for i in 1 2 3 4 5 6 7 8 9 10
        do
            python train_or_infer.py \
                --reconnection_frequency ${RECONNECTION_FREQUENCY} \
                infer \
                    --model_path /ubc/cs/home/s/shenranw/scratch/GNS/train_logs/rf=${RF}/run0/model.pth \
                    --logdir /ubc/cs/home/s/shenranw/scratch/GNS/rf=${RF}/rollouts
        done
        echo "------------------------------------------------------------------------------------------------"
    done
    echo "================================================================================================"
done
