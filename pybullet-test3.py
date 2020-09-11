import pybullet as p
import pybullet_data
from time import sleep
import numpy as np
import math
import utils_test

p.connect(p.GUI)
p.setGravity(0, 0, -10)

#床を作成
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("plane.urdf")
print("planeId:", planeId)#床のIDを表示
#車を出現
cubeStartPos = [0,0,0]
cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
carId=p.loadURDF("racecar/racecar.urdf",cubeStartPos, cubeStartOrientation)#racecarを出現
print("robotId:", carId)


#objectを作成
cube_path = 'C:/Users/yuki/PycharmProjects/pybullet/gym-pybullet/URDF/cube.urdf'
utils_test.make_object(urdf_path=cube_path, start_pos=[2,0,0], start_orientation=[0,0,0],obj_num=1)
utils_test.make_object(urdf_path="block.urdf", start_pos=[1,1,0], start_orientation=[0,0,1.57],obj_num=2)



#パラメータの作成
angle = p.addUserDebugParameter('Steering', -0.5, 0.5, 0)
throttle = p.addUserDebugParameter('Throttle', -20, 20, 0)
#動かすjointを指定
wheel_indices = [2, 3, 5, 7]
hinge_indices = [4, 6]

flag = True#衝突判定
t=0
while True:
    t +=1
    # パラメータの使用
    user_angle = p.readUserDebugParameter(angle)
    user_throttle = p.readUserDebugParameter(throttle)
    for joint_index in wheel_indices:
        p.setJointMotorControl2(carId, joint_index,
                                p.VELOCITY_CONTROL,
                                targetVelocity=user_throttle)
    for joint_index in hinge_indices:
        p.setJointMotorControl2(carId, joint_index,
                                p.POSITION_CONTROL,
                                targetPosition=user_angle)

    # 車の位置
    position, orientation = p.getBasePositionAndOrientation(carId)


    # 車の接触判定
    contact = p.getContactPoints(bodyA=1,bodyB=2)
    if len(contact) > 1:
        print("Ouch!", end=" ")
        flag = True

    #車載カメラ
    camera_pos, target_pos = utils_test.cal_camera_pos_and_target_pos(carId, 9)
    projectionMatrix = p.computeProjectionMatrixFOV(fov=120.0, aspect=1.0, nearVal=0.01, farVal=10.0)#fovは視野角
    viewMatrix = p.computeViewMatrix(camera_pos,target_pos,[0,0,1])
    p.getCameraImage(80, 80, viewMatrix,projectionMatrix)


    p.stepSimulation()

