import torch
import torch.nn as nn

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import sample
import random
from collections import namedtuple, deque

Experience = namedtuple('Experience',('state', 'action', 'next_state', 'reward'))

class Trainer:
    def __init__(self, args, env, memory, PolicyNet, TargetNet, optimizer, scheduler, device):
        super().__init__()
        self.args = args
        self.env = env
        self.memory = memory
        self.PolicyNet = PolicyNet
        self.TargetNet = TargetNet
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.steps_done = 0
        random.seed(self.args.seed_val)
        self.save_dir = os.path.join(self.args.save_path, self.args.experiment, '')

    def save_state(self, net, save_path):
        print('==> Saving models ...')
        state = {
                'net_state_dict': net.state_dict()
                }
        dir_name = '/'.join(save_path.split('/')[:-1])
        if not os.path.exists(dir_name):
            print("Creating Directory: ", dir_name)
            os.makedirs(dir_name)
        torch.save(state, str(save_path) + '.pth')

    def savePlots(self, losses, rewardVsEpisode):
        plt.figure()
        plt.plot(losses)
        plt.title("Loss vs Step")
        plt.savefig(self.save_dir+ 'LossVsStep.png')

        plt.figure()
        plt.plot(rewardVsEpisode)
        plt.title("Reward vs Episode")
        plt.savefig(self.save_dir+'RewardVsEpisode.png')

    def Images_to_Video(self, recordingFrames):
        vid_path = self.save_dir + 'video.mp4'
        frame_rate = 30
        h,w,c = recordingFrames[0].shape
        frame_size = (w,h)
        out = cv2.VideoWriter(vid_path,cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, frame_size)

        for i in range(len(recordingFrames)):
            recordingFrame = cv2.cvtColor(recordingFrames[i], cv2.COLOR_RGB2BGR)
            out.write(recordingFrame)
        out.release() 

    def check_path(self, fname):
        dir_name = '/'.join(fname.split('/')[:-1])
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    def select_action(self, state, areaOfObstacle):
        random_num = random.random()
        epsilon_thres = self.args.epsilon_stop + (self.args.epsilon_start - self.args.epsilon_stop)* \
                            np.exp(-1. * self.steps_done/self.args.epsilon_decay) 
        if random_num > epsilon_thres:
            # choose action via exploitation
            with torch.no_grad():
                q_values = self.PolicyNet(state, areaOfObstacle)
                _, action = torch.max(q_values, dim = 1)
                action = int(action.item())
        else:
            # choose action via exploration
            action = np.random.randint(self.args.num_actions)
        return action

    def optimisePolicyNet(self):
        if len(self.memory) < self.args.batch_size:
            return
        sampled_batch = self.memory.sampleBatch(self.args.batch_size)

        states = {'image': torch.stack([each[0]['image'] for each in sampled_batch], dim=1).squeeze(0).to(self.device), 
                'area_obstacle': torch.stack([each[0]['area_obstacle'] for each in sampled_batch], dim=1).squeeze(0).to(self.device)}
        actions = torch.tensor([each[1] for each in sampled_batch]).to(self.device).unsqueeze(-1)
        next_states = {'image': torch.stack([each[2]['image'] for each in sampled_batch], dim=1).squeeze(0).to(self.device), 
                'area_obstacle': torch.stack([each[2]['area_obstacle'] for each in sampled_batch], dim=1).squeeze(0).to(self.device)}
        rewards = torch.tensor([each[3] for each in sampled_batch]).to(self.device)

        state_action_values = self.PolicyNet(states['image'], states['area_obstacle'])
        state_action_values = state_action_values.gather(1, actions)
        next_state_values = torch.zeros(self.args.batch_size, device=self.device)
        next_state_values = self.TargetNet(next_states['image'], next_states['area_obstacle']).max(1)[0]
        episode_ends = []
        for i in range(next_states['image'].shape[0]):
            if torch.all(torch.eq(next_states['image'][i], torch.zeros(states['image'][0].shape).to(self.device))):
                episode_ends.append(i)
        episode_ends = torch.tensor(episode_ends)

        for i in episode_ends:
            next_state_values[i] = torch.tensor(0.0)
        expected_state_action_values = (next_state_values * self.args.gamma) + rewards

        criterion = nn.MSELoss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.PolicyNet.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        self.scheduler.step()
        return loss

    def train(self):
        save_outputfile = self.save_dir + 'output.txt'
        self.check_path(save_outputfile)
        f = open(save_outputfile, 'w')
        losses = []
        rewardVsEpisode = []
        for episode in range(1, self.args.num_episodes+1):
            total_reward = 0
            state, rgbImage = self.env.ImageAtCurrentPosition()
            recordingFrames = []
            recordingFrames.append(rgbImage)
            step = 0
            while True:
                self.steps_done+=1 # for epsilon_decay
                action = self.select_action(state['image'], state['area_obstacle'])
                done, next_state, reward, rgbImage = self.env.step(action)
                total_reward += reward
                self.memory.push((state, action, next_state, reward))
                recordingFrames.append(rgbImage)

                loss = self.optimisePolicyNet()
                losses.append(loss)
                if done or step == self.args.max_steps:
                    print('Episode: {}'.format(episode),'Total reward: {}'.format(total_reward[0][0].cpu().numpy()))

                    rewardVsEpisode.append(total_reward)
                    averageReward = torch.mean(torch.FloatTensor(rewardVsEpisode[episode-min(episode,self.args.reward_average_window):episode+1]))
                    f.write('Episode: {}, Total reward: {} \n'.format(episode, total_reward[0][0].cpu().numpy()))
                    self.env.reset()
                    break
                else:
                    state = next_state
                step+=1
            if episode % self.args.target_update == 0:
                self.TargetNet.load_state_dict(self.PolicyNet.state_dict())
            if averageReward >= self.args.averageRewardThreshold or episode == self.args.num_episodes:
                print("rewardVsEpisode")
                print(rewardVsEpisode)
                self.Images_to_Video(recordingFrames)
                self.savePlots(losses, rewardVsEpisode)
                self.save_state(self.PolicyNet, self.save_dir + self.args.experiment)
                break

        
