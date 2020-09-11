import gym
import torch
import gym_pybullet
from PPO_utils import PPOActor_pos
import numpy as np

weights_path = './weights/PPO_case6_500000steps.pth'

env = gym.make('pybullet_local-v2')
state_shape = env.observation_space_cordinate.shape
action_space = env.action_space.shape

print('observation space cordinate: ', state_shape)

print('action space: ',action_space )


net = PPOActor_pos(state_shape,action_space)
net_weights = torch.load(weights_path,map_location=torch.device('cpu'))
net.load_state_dict(net_weights)

states = env.reset()#今回の書き方では必要
print('goal:' + str(env.goal_pos_list))
done = False
while True:
  if done:
    states = env.reset()
    print('goal:' + str(env.goal_pos_list))
  states = torch.tensor(states)#staesをtorch.tensorに変換
  action = net(states)
  states, reward, done, info = env.step(action)
