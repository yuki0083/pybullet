import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pybullet as p
import pybullet_data
import sys
import os
#sys.path.append('..')#pythonがファイルを検索するときに使うパスに親ディレクトリを追加(utils.pyのため)
from ..import env_utils
import numpy as np
import random


class Pybullet_env(gym.Env):
    def __init__(self):
        #or p.DIRECT for non-graphical version
        #p.connect(p.DIRECT)
        p.connect(p.GUI)
        p.setGravity(0, 0, -10)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        #set_action_space
        self.min_throttle = -20
        self.max_throttle = 20
        self.min_angle = -0.5
        self.max_angle = 0.5
        self.action_low = np.array([self.min_throttle, self.min_angle])#min(max)をndarray(2,)で書くことで行動空間も(2,)になる
        self.action_high = np.array([self.max_throttle, self.max_angle])
        self.action_space = spaces.Box(self.action_low, self.action_high, dtype=np.float32)#連続値の(2次元行動)空間を作る

        #set_observation_space
        self.observation_space_camera = spaces.Box(low=0,high=255,shape=(3,80,80,3))#shape=(3,80,80,3)?
        self.observation_space_cordinate = spaces.Box(low=-15,high=15, shape=(2,))#low=-15でいいのか?

        # carのプロパティ
        #self.carId = 0
        self.wheel_indices = [2, 3, 5, 7]# 動かすjointを指定
        self.hinge_indices = [4, 6]
        #self.position = [0,0,0]#初期位置
        #self.orientation = p.getQuaternionFromEuler([0, 0, 0])#初期クオータニオン

        #オブジェクトの位置の設定
        self.num_of_objects = 3
        #self.obj_poss_list = utils.make_obj_poss_list(self.num_of_objects,self.map_size)#objectの位置のリスト

        #報酬の設定
        self.goal_reward = 1000
        self.collision_reward = -100
        #1epidodeでの最大step数
        self.max_step_num = 1000
        #mapの一辺の大きさ
        self.map_size = 5

    # actionを実行し、結果を返す
    def step(self,actions):
        self.step_num += 1
        action_throttle,action_angle = actions#actionsはタイヤの速度と角度の2要素のリストを想定
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
        reward = self.get_reward()
        state = self.get_observation()
        done = self.done
        """
        if (self.is_goal() == True) or (self.is_collision()==True)or(self.step_num>self.max_step_num):#1stepごとに行動を選択
            done = True
        p.stepSimulation()
        """

        for _ in range(100):#100stepごとに行動を更新し、報酬や状態を返す
            if (self.is_goal() == True) or (self.is_collision() == True) or (self.step_num > self.max_step_num):
                done = True
            p.stepSimulation()

        return state, reward, done

    #状態を返す
    def get_observation(self):
        #カメラ画像を返す
        camera_pos, target_pos = env_utils.cal_camera_pos_and_target_pos(self.carId, 9)
        projectionMatrix = p.computeProjectionMatrixFOV(fov=120.0, aspect=1.0, nearVal=0.01, farVal=10.0)  # fovは視野角
        viewMatrix = p.computeViewMatrix(camera_pos, target_pos, [0, 0, 1])
        width, height, rgb, depth, segmentation = p.getCameraImage(80, 80, viewMatrix, projectionMatrix)
        camera_img_list = [rgb,depth,segmentation]
        #車から見た目標地点までの相対位置を返す
        relative_goal_pos_list = (np.array(self.goal_pos_list)-np.array(self.position[:-1])).tolist()

        return camera_img_list, relative_goal_pos_list

    # 状態を初期化し、初期の観測値を返す
    def reset(self):
        self.step_num = 0 #episode内でのstep数
        self.done = False#終了判定
        p.resetSimulation()#シミュレーション環境のリセット
        p.setGravity(0, 0, -10)#resetした後、重力再設定
        self.planeId = p.loadURDF("plane.urdf")#床の読み込み

        #目標位置の設定
        self.goal_pos_list = env_utils.set_random_point(self.map_size)#(x,y)


        #車の読み込み
        self.position = [0,0,0]#初期位置
        self.orientation = p.getQuaternionFromEuler([0, 0, 0])#初期クオータニオン
        self.carId = p.loadURDF("racecar/racecar.urdf", self.position, self.orientation)

        #objectの読み込み
        self.obj_id_list = []#obj_idを格納
        self.obj_poss_list = env_utils.make_obj_poss_list(self.num_of_objects, self.map_size)  # objectの位置のリストを取得
        for obj_poss_list in self.obj_poss_list:
            obj_poss_list.append(0)#(x,y,0)とする
            obj_orientation = [0, 0, random.uniform(0,6.28)]#yowをランダムに選択
            ob_id = env_utils.make_object(urdf_path="block.urdf", start_pos=obj_poss_list, start_orientation=obj_orientation)
            self.obj_id_list.append(ob_id)

        #p.stepSimulation()
        #reward = self.get_reward

    #報酬値を返す
    def get_reward(self):
        if self.is_goal():#goalした時
            return self.goal_reward
        elif self.is_collision():#接触した時
            return self.collision_reward
        else:
            return -1 #1stepごとに-1

    #goal判定
    def is_goal(self):
        flag = False
        self.error = 0.1 #positionの許容範囲
        if (self.goal_pos_list[0]-self.error<self.position[0]<self.goal_pos_list[0]+self.error)\
                and (self.goal_pos_list[1]-self.error<self.position[1]<self.goal_pos_list[1]+self.error):
            flag = True
        return flag

    #レンダリング
    def render(self, mode='human', close=False):
        """
        #colabの場合
        width, height, rgb, depth, segmentation = p.getCameraImage(360,240)
        return rgb
        """
        #ローカルの場合
        ...

    # 衝突判定
    def is_collision(self):
        flag = False
        for obj_id in  self.obj_id_list:
            contact = p.getContactPoints(bodyA=self.carId, bodyB=obj_id)
            if len(contact) > 1:
                #print("Ouch!", end=" ")
                flag = True
        return flag

    #環境を閉じる
    def close(self):
        p.disconnect()





