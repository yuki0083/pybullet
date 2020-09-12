from gym.envs.registration import register

register(
    id='pybullet-v0',
    entry_point='gym_pybullet.envs:Pybullet_env',#pybullet_env.pyの中のPybullet_envクラス
)                   #__ini__.pyでPybullet_envをimportしているのでpybullet_envは書かなくていい

register(
    id='pybullet_local-v0',
    entry_point='gym_pybullet.envs:Pybullet_env_local',
)


register(
    id='pybullet-v2',
    entry_point='gym_pybullet.envs:Pybullet_env2',#pybullet_env2.pyの中のPybullet_env2クラス
)                   #__ini__.pyでPybullet_env2をimportしているのでpybullet_env2は書かなくていい

register(
    id='pybullet_local-v2',
    entry_point='gym_pybullet.envs:Pybullet_env2_local',
)

register(
    id='pybullet_handle-v0',
    entry_point='gym_pybullet.envs:Pybullet_env_handle',
)

register(
    id='pybullet_handle_local-v0',
    entry_point='gym_pybullet.envs:Pybullet_env_handle_local',
)