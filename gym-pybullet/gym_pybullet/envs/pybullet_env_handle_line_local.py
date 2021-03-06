import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pybullet as p
import pybullet_data
import sys
import os
# sys.path.append('..')#pythonがファイルを検索するときに使うパスに親ディレクトリを追加(utils.pyのため)
from .. import env_utils
import numpy as np
import random


class Pybullet_env_handle_line_local(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # or p.DIRECT for non-graphical version
        #p.connect(p.DIRECT)
        p.connect(p.GUI)
        p.setGravity(0, 0, -10)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # set_action_space
        # self.min_throttle = -20
        # self.max_throttle = 20
        self.action_throttle = 5
        self.min_angle = -0.5
        self.max_angle = 0.5
        self.action_low = np.array([self.min_angle])  # min(max)をndarray(2,)で書くことで行動空間も(2,)になる
        self.action_high = np.array([self.max_angle])
        self.action_space = spaces.Box(self.action_low, self.action_high, dtype=np.float32)  # 連続値の(2次元行動)空間を作る

        # set_observation_space
        self.observation_space_camera = spaces.Box(low=-100, high=100, shape=(
        84, 84, 1))  # shape=(3,84,84,3)?　#segmentationのみにした(84,84,1) ###################################
        self.observation_space_cordinate = spaces.Box(low=-100, high=100, shape=(2,))  # low=-15でいいのか?

        # carのプロパティ
        # self.carId = 0
        self.wheel_indices = [2, 3, 5, 7]  # 動かすjointを指定
        self.hinge_indices = [4, 6]
        # self.position = [0,0,0]#初期位置
        # self.orientation = p.getQuaternionFromEuler([0, 0, 0])#初期クオータニオン

        # オブジェクトの位置の設定
        self.num_of_objects = 5
        # self.obj_poss_list = utils.make_obj_poss_list(self.num_of_objects,self.map_size)#objectの位置のリスト

        # 報酬の設定
        self.goal_reward = 1
        self.collision_reward = -0.5

        # 1epidodeでの最大step数
        self._max_episode_steps = 10000

        # mapの一辺の大きさ
        # self.map_size = 6
        self.map_size_x = 4
        self.map_size_y = 2

    # actionを実行し、結果を返す
    def step(self, actions):
        # p.connect(p.DIRECT)
        self.step_num += 1
        # action_throttle,action_angle = actions#actionsはタイヤの速度と角度の2要素のリストを想定
        action_throttle = self.action_throttle
        action_angle = actions
        for joint_index in self.wheel_indices:
            p.setJointMotorControl2(self.carId, joint_index,
                                    p.VELOCITY_CONTROL,
                                    targetVelocity=action_throttle)
        for joint_index in self.hinge_indices:
            p.setJointMotorControl2(self.carId, joint_index,
                                    p.POSITION_CONTROL,
                                    targetPosition=action_angle)
        # 車の位置を取得
        self.position, self.orientation = p.getBasePositionAndOrientation(self.carId)

        state = self.get_observation()

        self.time_reward = -(self.map_size_x / 2 + 1 - self.position[0]) / 100  # ゴールまでの距離を報酬にする

        reward = self.get_reward()
        #print(reward)

        done = self.done
        """
        if (self.is_goal() == True) or (self.is_collision()==True)or(self.step_num>self.max_step_num):#1stepごとに行動を選択
            done = True
        p.stepSimulation()
        """

        for _ in range(3):  # 100stepごとに行動を更新し、報酬や状態を返す
            if self.is_goal() == True:
                done = True
                reward = self.goal_reward
                break
            elif self.is_collision() == True:
                #done =True #当たっても終わりにしない
                reward = self.collision_reward
                #break
            """elif self.step_num > self._max_episode_steps:
                done = True
                break"""

            """if (self.is_goal() == True) or (self.is_collision() == True) or (self.step_num > self._max_episode_steps):
                done = True
            """
            p.stepSimulation()

        if self.step_num > self._max_episode_steps:
            done = True

        return state, reward, done, {}

    # 状態を返す
    def get_observation(self):
        # カメラ画像を返す
        camera_pos, target_pos = env_utils.cal_camera_pos_and_target_pos(self.carId, 9)
        projectionMatrix = p.computeProjectionMatrixFOV(fov=120.0, aspect=1.0, nearVal=0.01, farVal=10.0)  # fovは視野角
        viewMatrix = p.computeViewMatrix(camera_pos, target_pos, [0, 0, 1])
        width, height, rgb, depth, segmentation = p.getCameraImage(84, 84, viewMatrix,
                                                                   projectionMatrix)  #########################################################
        # camera_img_list = [rgb,depth,segmentation] segmentaionのみに変更
        # 車から見た目標地点までの相対位置を返す
        # relative_goal_pos_list = (np.array(self.goal_pos_list)-np.array(self.position[:-1])).tolist()
        # 自分の位置を状態とする
        self_pos_list = self.position[:-1]

        return [segmentation, self.position[:-1]]

    # 状態を初期化し、初期の観測値を返す
    def reset(self):
        # p.connect(p.DIRECT)
        self.step_num = 0  # episode内でのstep数
        self.done = False  # 終了判定
        p.resetSimulation()  # シミュレーション環境のリセット
        p.setGravity(0, 0, -10)  # resetした後、重力再設定
        self.planeId = p.loadURDF("plane.urdf")  # 床の読み込み

        # 目標位置の設定
        # self.goal_pos_list = env_utils.set_random_lattice_point(self.map_size)#(x,y)
        self.goal_x = 5  # x座標5がgoal

        # 車の読み込み
        self.position = [-3, 0, 0]  # 初期位置
        self.orientation = p.getQuaternionFromEuler([0, 0, 0])  # 初期クオータニオン
        self.carId = p.loadURDF("racecar/racecar.urdf", self.position, self.orientation)

        # objectの読み込み
        self.obj_id_list = []  # obj_idを格納
        self.obj_poss_list = env_utils.make_obj_lattice_poss_list_for_line(self.num_of_objects, self.map_size_x,
                                                                           self.map_size_y)  # objectの位置のリストを取得
        for obj_poss_list in self.obj_poss_list:
            obj_poss_list.append(0.18)  # (x,y,0)とする
            obj_orientation = [0, 0, 0]  # yowをランダムに選択
            ob_id = env_utils.make_object(urdf_path="./URDF/cube.urdf", start_pos=obj_poss_list,
                                          start_orientation=obj_orientation)
            self.obj_id_list.append(ob_id)

        # 壁の作成
        wall_objId = env_utils.make_wall_for_line("./URDF/wall.urdf", self.map_size_y + 2)
        for i in range(2):
            self.obj_id_list.append(wall_objId[i])

        # p.stepSimulation()
        # reward = self.get_reward

        state = self.get_observation()  # resetでもstateを返す
        return state

    # 報酬値を返す
    def get_reward(self):
        if self.is_goal():  # goalした時
            return self.goal_reward
        elif self.is_collision():  # 接触した時
            return self.collision_reward
        else:
            return self.time_reward  # 1stepごとにゴールまでの距離

    # goal判定
    def is_goal(self):
        flag = False
        if self.position[0] > self.map_size_x / 2 + 1:
            flag = True
        return flag

    # レンダリング
    def render(self, mode='human', close=False):
        """
        #colabの場合
        width, height, rgb, depth, segmentation = p.getCameraImage(360,240)
        return rgb
        """
        # ローカルの場合
        ...

    # 衝突判定
    def is_collision(self):
        flag = False
        for obj_id in self.obj_id_list:
            contact = p.getContactPoints(bodyA=self.carId, bodyB=obj_id)
            if len(contact) > 1:
                # print("Ouch!", end=" ")
                flag = True
        return flag

    # 環境を閉じる
    def close(self):
        p.disconnect()





