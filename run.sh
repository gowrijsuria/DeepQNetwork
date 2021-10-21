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

python3 main.py --save_path /ssd_scratch/cvit/gowri/DQN/ --experiment oct21_batch32_episodes150_steps20_coll_half_avgwind_5 --max_steps 20 --reward_average_window 5 --num_episodes 150 --averageRewardThreshold 4500