import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pybullet as p
from typing import Union
import pdb
import os.path as osp

from pb_diff_envs.utils.kuka_utils_luo import NP_RED, NP_GREEN, NP_PINK,NP_GREEN_2, check_start_end, set_collision_marker, set_collision_frame


# import multiprocessing as mp
# vis_load_env_load = mp.Lock() # useless, should use the same locks
def robogroup_visualize_traj_luo(env, traj, lock=None, is_ee3d=False, is_debug=False):
    '''
    *The same design as visualize_kuka_traj_luo*
    This function should support all the robot group (e.g., triple robots/ 6d arms) 
    The difference with visualize_traj is that this function directly receives 
    Args:
    [a list of 1d numpy] or [2d np] as trajectory, not using a traj_agent

    '''
    ## the objects' trajectories, should be 2 for static obj
    max_len_traj = max([len(obj.trajectory.waypoints) for obj in env.objects])
    gifs = []
    max_len = max(len(traj), max_len_traj)
    assert max_len == len(traj)
    print('[vis traj] max_len:', max_len)

    se_valid = check_start_end(traj, thres=0.5*env.robot.num_robots) # if the start and end is valid
    # if not p.isConnected():
    if lock is not None:
        lock.acquire()
        env.load(GUI=False)
        lock.release()
    else:
        # if p.isConnected()
        # print(f'p isConnected: {p.isConnected()}')
        env.load(GUI=is_debug) # ok

    i_y, i_p = 0, 0
    img_w, img_h = env.img_w, env.img_h



    color_1 = plt.cm.jet(np.linspace(0,1, max_len)) # color of the dot
    color_2 = plt.cm.magma(np.linspace(0,1, max_len)) 
    color_3 = plt.cm.viridis(np.linspace(0,1, max_len))
    colors = [color_1, color_2, color_3] # three max 3 robots

    vshape_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.02)  # Red color
    has_collision = False
    num_collision_frame = 0
    collision_list = []
    robogroup_add_start_end_marker(env, traj)
    assert not is_ee3d, 'depricated, prevent bugs'
    # for c, timestep in tqdm(enumerate(traj)):
    for i_t in tqdm(range(max_len)):

        if not is_ee3d: # traj is xyz-level
            env.robot.set_config(traj[i_t])
            new_pos_list = env.robot.get_end_effector_loc3d() # np 2,3
        else:
            raise NotImplementedError()

        # cam_pos = p.getLinkState(env.robot_id, 5)[0]
        # new_pos_q = p.getQuaternionFromEuler(new_pos)

        # dist = 2 - np.linalg.norm(cam_pos)

        ## add current frame ee loc
        for i_r in range(len(new_pos_list)):
            tmp_id = p.createMultiBody(baseMass=0, basePosition=new_pos_list[i_r], baseVisualShapeIndex=vshape_id,)
            p.changeVisualShape(tmp_id, linkIndex=-1, rgbaColor=colors[i_r][i_t])


        # print(p.getBodyInfo(tmp_id))
        no_collision = env.robot.no_collision()
        if not no_collision:
            has_collision = True
            num_collision_frame += 1
            collision_list.append(i_t)
            # print(f'{i_t} has_collision {has_collision};')





        cam_pos = [0,0,0.3]
        dist = 1.8
        
        view_mat = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=cam_pos, distance=dist, # [0, 0, 0]
                yaw=-i_y, pitch=-40-i_p, roll=0, upAxisIndex=2) # yaw=-90-i_y
        i_p = i_p + 0.2 if i_p < 15 else i_p
        i_y += 0.3
        

        gifs.append(p.getCameraImage(width=img_w, height=img_h, lightDirection=[1, 1, 1], shadow=0,
                                                 renderer=p.ER_TINY_RENDERER,
                                                 viewMatrix=view_mat,
                                                 projectionMatrix=env.proj_mat,
                                                 flags=p.ER_NO_SEGMENTATION_MASK,
                                                 )[2])
    
    
    end_yaw = 40 + i_y
    gifs.extend([gifs[-1]]*10)
    rot_gap = 9; n_rot_frames = round(360 / rot_gap)
    for i_y in range(0, 360, rot_gap): # 2 3
        view_mat = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=cam_pos, distance=dist, # [0, 0, 0]
                yaw=(end_yaw+i_y), pitch=-50, roll=0, upAxisIndex=2)
        gifs.append(p.getCameraImage(width=img_w, height=img_h, lightDirection=[1, 1, 1], shadow=0,
                                                #  renderer=p.ER_BULLET_HARDWARE_OPENGL,
                                                 renderer=p.ER_TINY_RENDERER,
                                                 viewMatrix=view_mat,
                                                 projectionMatrix=env.proj_mat,
                                                 flags=p.ER_NO_SEGMENTATION_MASK,
                                                 )[2])

    gifs = set_collision_marker(gifs, has_collision, se_valid)
    set_collision_frame(gifs, collision_list)
    print(f'[robogroup vis traj] num_collision_frame: {num_collision_frame}; start end valid: {se_valid}')



    ds = [100] * len(gifs)
    ds[-n_rot_frames:] = [300,] * n_rot_frames # slow down
    ds[0] = 1200
    ds[-1] = 5000
    vis_dict = dict(ncoll_frame=num_collision_frame, se_valid=se_valid)
    env.unload_env()
    return gifs, ds, vis_dict


def robogroup_add_start_end_marker(env, traj):
    '''add special marker at the start and end position'''

    env.robot.set_config(traj[0])
    start_pos_list = env.robot.get_end_effector_loc3d() # np 2,3

    env.robot.set_config(traj[-1])
    end_pos_list = env.robot.get_end_effector_loc3d() # np 2,3

    # create two larger ball in two ends
    start_vid = p.createVisualShape(p.GEOM_SPHERE, radius=0.06)  # Red color
    for start_pos in start_pos_list:
        tmp_id = p.createMultiBody(baseMass=0, basePosition=start_pos, baseVisualShapeIndex=start_vid,)
        p.changeVisualShape(tmp_id, linkIndex=-1, rgbaColor=NP_GREEN_2)

    end_vid = p.createVisualShape(p.GEOM_SPHERE, radius=0.06)  # Red color
    for end_pos in end_pos_list:
        tmp_id = p.createMultiBody(baseMass=0, basePosition=end_pos, baseVisualShapeIndex=end_vid,)
        p.changeVisualShape(tmp_id, linkIndex=-1, rgbaColor=NP_PINK)

def robotgroup_check_start_end(traj, thres=0.5): # 0.2
    '''check if trajectory is close to the start and end'''
    raise NotImplementedError('no need for now')
