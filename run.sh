#!/bin/bash
#SBATCH -A gowri
#SBATCH -n 20
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
#SBATCH --mincpus=20
#SBATCH --mail-type=END
##SBATCH --nodelist=node01

module add cuda/10.1
module add cudnn/7.6.2.24-cuda-10.1

python3 main.py --batch_size 32