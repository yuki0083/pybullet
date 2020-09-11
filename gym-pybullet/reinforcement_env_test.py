import gym
import gym_pybullet
env = gym.make('pybullet_local-v2')
#print('observation space camera: ', env.observation_space_camera.shape)
print('observation space cordinate: ', env.observation_space_cordinate.shape[0])

print('action space: ', env.action_space)

env.reset()#今回の書き方では必要
done = False
while True:
  if done:
    env.reset()
  action = env.action_space.sample()
  state, reward, done, info = env.step(action)
