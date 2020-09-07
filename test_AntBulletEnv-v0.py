import gym
import pybullet
import pybullet_envs

# 環境の作成
env = gym.make('AntBulletEnv-v0')

# ランダム行動による動作確認
env.render(mode='human')
env.reset()
while True:
    # 1ステップ実行
    state, reward, done, info = env.step(env.action_space.sample())
    print('reward:', reward)
    # エピソード完了
    if done:
        print('done')
        state = env.reset()