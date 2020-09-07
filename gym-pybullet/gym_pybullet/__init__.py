from gym.envs.registration import register

register(
    id='pybullet-v0',
    entry_point='gym_pybullet.envs:Pybullet_env',#pybullet_env.pyの中のPybullet_envクラス
)                   #__ini__.pyでPybullet_envをimportしているのでpybullet_envは書かなくていい