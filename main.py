import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import pybullet as p
import pybullet_data
import time
import argparse
import os

from train import Trainer
from network import DQNetwork
from data import ReplayMemory
from environment import CarEnvironment


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', action='store', type=int, default=32)
parser.add_argument('--workers',    action='store', type=int, default=6)
parser.add_argument('--seed_val',    action='store', type=int, default=1)

# Model parameters
parser.add_argument('--num_episodes',            action='store', type=int, default=100)
parser.add_argument('--max_steps',               action='store', type=int, default=100)
parser.add_argument('--num_actions',             action='store', type=int, default=4) # 0-L, 1-R, 2-F, 3-B
parser.add_argument('--replay_memory_size',      action='store', type=int, default=10000)
parser.add_argument('--gamma',                   action='store', type=float, default=0.999, help='future reward discount')
parser.add_argument('--epsilon_start',           action='store', type=float, default=0.9, help='start epsilon')
parser.add_argument('--epsilon_stop',            action='store', type=float, default=0.05, help='stop epsilon')
parser.add_argument('--epsilon_decay',           action='store', type=int, default=250, help='epsilon decay')
parser.add_argument('--threshold_dist',          action='store', type=float, default=2)
parser.add_argument('--input_channels',          action='store', type=int, default=3)
parser.add_argument('--target_update',           action='store', type=int, default=10)
parser.add_argument('--reward_average_window',   action='store', type=int, default=10)
parser.add_argument('--averageRewardThreshold',  action='store', type=int, default=4500)

# Environment parameters
parser.add_argument('--start_x',                 action='store', type=int, default=0)
parser.add_argument('--start_y',                 action='store', type=int, default=0)
parser.add_argument('--start_z',                 action='store', type=float, default=0.1)
parser.add_argument('--final_x',                 action='store', type=int, default=-3)
parser.add_argument('--final_y',                 action='store', type=int, default=8)
parser.add_argument('--final_z',                 action='store', type=int, default=0)
parser.add_argument('--reward_goal',             action='store', type=int, default=5000)
parser.add_argument('--reward_collision',        action='store', type=float, default=-0.05)
parser.add_argument('--reward_outside_boundary', action='store', type=int, default=-100)
parser.add_argument('--screen_height',           action='store', type=int, default=200)
parser.add_argument('--screen_width',            action='store', type=int, default=200)

# Agent parameters
parser.add_argument('--targetVel', action='store', type=int, default=5, help='rad/s')
parser.add_argument('--maxForce',  action='store', type=int, default=50)

# Default parameters
parser.add_argument('--lr',         action='store', type=float,default=0.001)
parser.add_argument('--step_size',  action='store', type=float,default=20)
parser.add_argument('--LR_gamma',   action='store', type=float,default=0.1)
parser.add_argument('--experiment', action='store', type=str,default='trial')
parser.add_argument('--save_path',  action='store', type=str,default='./checkpoints/')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PolicyNet = DQNetwork(args.input_channels, args.num_actions).to(device)
TargetNet = DQNetwork(args.input_channels, args.num_actions).to(device)
TargetNet.load_state_dict(PolicyNet.state_dict())
TargetNet.eval()

optimizer = optim.Adam(PolicyNet.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 40, 80], gamma=args.LR_gamma)
  
memory = ReplayMemory(args, args.replay_memory_size)
env = CarEnvironment(args, device)
env.reset()

trainer = Trainer(args, env, memory, PolicyNet, TargetNet, optimizer, scheduler, device)
trainer.train()

print("Training Complete!")
env.close()
