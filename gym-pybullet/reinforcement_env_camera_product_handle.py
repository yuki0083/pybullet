import gym
import torch
import gym_pybullet
from PPO_utils import PPOActor_camera
import numpy as np

weights_path = './weights/PPO_camera_handle50000steps.pth'

env = gym.make('pybullet_handle_local-v0')
state_pos_shape = env.observation_space_cordinate.shape
state_camera_shape = env.observation_space_camera.shape
action_space = env.action_space.shape

print('observation camera space :',state_camera_shape)
print('observation pos space : ', state_pos_shape)

print('action space: ',action_space )

state_shape =[state_camera_shape, state_pos_shape]

net = PPOActor_camera(state_shape,action_space)
net_weights = torch.load(weights_path,map_location=torch.device('cpu'))
net.load_state_dict(net_weights)

states = env.reset()#今回の書き方では必要
print('goal:' + str(env.goal_pos_list))
done = False
while True:
  if done:
    states = env.reset()
    print('goal:' + str(env.goal_pos_list))
  cam_states = torch.tensor(states[0],dtype=torch.float32).unsqueeze(0).unsqueeze(0)#staesをtorch.tensorに変換
  pos_states = torch.tensor(states[1]).unsqueeze(0)
  action = net(cam_states,pos_states)
  states, reward, done, info = env.step(action)
