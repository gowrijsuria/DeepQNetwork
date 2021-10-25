#!/bin/bash
#SBATCH -A gowri
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
#SBATCH --mincpus=10
#SBATCH --mail-type=END
#SBATCH --nodelist=gnode03

module add cuda/10.2
module add cudnn/7.6.5-cuda-10.2

python3 main.py --save_path /ssd_scratch/cvit/gowri/DQN/ --experiment oct25_batch32_episodes500_steps50_coll0_01_avgwind5_maxForce_40 --max_steps 50 --reward_average_window 5 --num_episodes 500 --averageRewardThreshold 4500 --threshold_dist 2 --batch_size 32 \
        --targetVel 5 \
        --maxForce 40 \
        --reward_collision -0.01

