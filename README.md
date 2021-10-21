# DeepQNetwork
Implementation of Deep Q Network in Pybullet

## Objective
Use DeepQNetworks to find the optimal path to the destination using an environment created using pybullet. The environment contains obstacles and a goal point(soccerball). The input states to the network are images and the total area of obstacles in the scene (found using segmentation masks).

<img width="1142" alt="Screen Shot 2021-10-21 at 1 00 49 PM" src="https://user-images.githubusercontent.com/32260835/138304509-ff2a5776-83d9-4462-a473-aa41001a1faf.png">

### Hyperparameters Used
● Input State: image captured + total area of all obstacles obtained using
segmentation masks
● Possible Actions - 4 directions (Left, Right, Front, Back)
● Reward
○ On reaching goal = 5000
○ On collliding with obstacles = -0.5*area of obstacles
○ Outside boundary = -100
● Threshold distance to goal = 2
● Episodes = 150
● Max_Steps = 20
● Replaymemory size = 10000
● Epsilon_start = 0.9
● Epsilon_stop = 0.05
