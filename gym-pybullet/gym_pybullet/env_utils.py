import math
import pybullet as p
import numpy as np
import pybullet_data
import random

def cal_target_pos(cam_pos,cam_vec):#target_posを計算(cal_camera_pos_and_target_posの一部)
    np_cam_position = np.array(list(cam_pos))
    np_cam_vec = np.array(cam_vec)
    target_pos = (np_cam_position +np_cam_vec).tolist()
    return target_pos

#carIdとカメラのlinkのidからcamera_posと(カメラの)target_posを返す
def cal_camera_pos_and_target_pos(carId,link_id):
    camera_pos =p.getLinkState(carId, link_id)[0]
    camera_orientation = p.getLinkState(carId,9)[1]
    euler_angles = p.getEulerFromQuaternion(camera_orientation)
    yow_angle = euler_angles[2]
    camera_vec = [math.cos(yow_angle),math.sin(yow_angle),0]
    target_pos = cal_target_pos(camera_pos,camera_vec)
    return camera_pos,target_pos

#objectを作成
def make_object(urdf_path, start_pos, start_orientation, globalScaling=20):
    cubeStartPos = start_pos
    cubeStartOrientation = p.getQuaternionFromEuler(start_orientation)
    obId = p.loadURDF(urdf_path, cubeStartPos, cubeStartOrientation, globalScaling=globalScaling)
    return obId
    #print("ob{}Id:".format(obj_num), obId)

def make_wall(urdf_path, map_size):
    obId_list = []
    wall_z_cordinate = 0.18
    wall_pos_list = [[0,(map_size+2)/2,wall_z_cordinate], [0,-(map_size+2)/2,wall_z_cordinate], [(map_size+2)/2,0,wall_z_cordinate], [-(map_size+2)/2,0,wall_z_cordinate]]
    cubeStartOrientation1 = p.getQuaternionFromEuler([0, 0, 0])
    cubeStartOrientation2 = p.getQuaternionFromEuler([0, 0, 1.57])
    for i in range(2):
        cubeStartPos = wall_pos_list[i]
        obId = p.loadURDF(urdf_path,cubeStartPos,cubeStartOrientation1,globalScaling=20)
        obId_list.append(obId)
    for i in [2,3]:
        cubeStartPos = wall_pos_list[i]
        obId = p.loadURDF(urdf_path, cubeStartPos, cubeStartOrientation2,globalScaling=20)
        obId_list.append(obId)

    return obId_list

#マップ内で座標をランダムに設定する関数
def set_random_point(map_size):
    x_cordinate = random.uniform(-map_size / 2, map_size / 2)
    y_cordinate = random.uniform(-map_size / 2, map_size / 2)
    return [x_cordinate, y_cordinate]#出力はリスト(x,y)

#オブジェクトの座標(x,y)のリストを作成する関数
def make_obj_poss_list(num_of_objects,map_size):
    obj_poss_list = []
    for _ in range(num_of_objects):
        obj_pos_list = set_random_point(map_size)
        obj_poss_list.append(obj_pos_list)
    return obj_poss_list

def set_random_lattice_point(map_size):#格子点上の座標を取得(-1,0,1)は除く
    out_points = [0]
    while True:
        x_cordinate = random.randint(-map_size / 2, map_size / 2)
        if not x_cordinate in out_points:
            break
    while True:
        y_cordinate = random.randint(-map_size / 2, map_size / 2)
        if not x_cordinate in out_points:
            break
    return [x_cordinate, y_cordinate]


def make_obj_lattice_poss_list(num_of_objects,map_size,goal_pos):#オブジェクトの座標(格子点上)
    obj_poss_lists = []
    while True:
        obj_pos_list = set_random_lattice_point(map_size)
        if (obj_pos_list != goal_pos)and(obj_pos_list not in obj_poss_lists):
            obj_poss_lists.append(obj_pos_list)
        if len(obj_poss_lists)==num_of_objects:
            break
    return obj_poss_lists

#pybullet_env_handle_line用の関数
def set_random_lattice_point_for_line(map_size_x,map_size_y):
        x_cordinate = random.randint(-map_size_x / 2, map_size_x / 2)
        y_cordinate = random.randint(-map_size_y / 2, map_size_y / 2)
        return [x_cordinate, y_cordinate]




def make_obj_lattice_poss_list_for_line(num_of_objects,map_size_x,map_size_y):#オブジェクトの座標(格子点上)
    obj_poss_lists = []
    while True:
        obj_pos_list = set_random_lattice_point_for_line(map_size_x,map_size_y)
        if obj_pos_list not in obj_poss_lists:
            obj_poss_lists.append(obj_pos_list)
        if len(obj_poss_lists)==num_of_objects:
            break
    return obj_poss_lists


def make_wall_for_line(urdf_path, lines_distance):
    obId_list = []
    wall_z_cordinate = 0.18
    wall_pos_list = [[0, lines_distance/ 2, wall_z_cordinate], [0, -lines_distance / 2, wall_z_cordinate]]
    cubeStartOrientation1 = p.getQuaternionFromEuler([0, 0, 0])
    for i in wall_pos_list:
        cubeStartPos = i
        obId = p.loadURDF(urdf_path, cubeStartPos, cubeStartOrientation1, globalScaling=20)
        obId_list.append(obId)

    return obId_list
