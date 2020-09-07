import gym
import gym_pybullet
env = gym.make('pybullet-v0')
print('observation space camera: ', env.observation_space_camera)
print('observation space cordinate: ', env.observation_space_cordinate)
print('action space: ', env.action_space)

env.reset()#今回の書き方では必要
done = False
while (not done):

  action = env.action_space.sample()
  state, reward, done, info = env.step(action)
