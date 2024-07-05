import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pybullet as p
from typing import Union
import pdb
import os.path as osp


class SolutionInterp:
    def __init__(self, density=10) -> None:
        self.density = density
        
    def __call__(self, solution: Union[np.ndarray, list], return_dict=False):
        '''
        returns 
        sols: a numpy array (n,7) or a list of np(7,)
        '''
        if type(solution) == np.ndarray:
            assert solution.ndim == 2
        sol_len = len(solution)
        sol_ori_idx = [0,]
        sols = []

        for i in range(sol_len - 1):
            diff = solution[i + 1] - solution[i]
            diff_norm = np.linalg.norm(diff)
            diff_abs_sum = np.abs(diff).sum()

            if diff_norm > 1e-3:
                # n_insert = np.ceil(diff_norm * self.density).astype(np.int32) # density 10
                n_insert = np.ceil(diff_abs_sum * self.density / 2).astype(np.int32) # density 5
            else:
                n_insert = 0

            # print(diff_norm, n_insert, diff_abs_sum)

            x = np.linspace(0, 1, n_insert + 2).reshape(-1, 1) # + 2 for start and end points
            
            sol_ori_idx.append(sol_ori_idx[-1] + len(x) - 1)

            if i != (sol_len - 2):
                # remove the last elem, prevent duplicate [x_1, x_2)
                x = x[:-1]


            # (n,1) * (1,n)
            sol_interp = solution[i] + x * diff.reshape(1, -1)
            sols.append(sol_interp)
            # print('sol_interp[(0,1,-1)]:', sol_interp[(0,1,-1),])
        

        if not return_dict:
            return np.concatenate(sols)
        else:
            sols = np.concatenate(sols)
            is_sol_keypoint = np.zeros(shape=sols.shape[0], dtype=bool)
            is_sol_keypoint[sol_ori_idx] = True
            infos = dict(is_sol_keypoint=is_sol_keypoint)
            # print(f'sols {len(sols)}', 'sol_ori_idx:', sol_ori_idx) # [0, 52, 87]
            return sols, infos

if __name__ == '__main__':
    result_tmp = [
            np.array([ 2.64058395,  1.4584981 , -1.59389355, -0.60844031]), 
            np.array([ 0.94870739,  0.53262687, 1.02528032, -3.05241936]),  
            np.array([-2.0180678, -1.5233936, -0.85391584, -1.8498511 ])
            ]
    sol_interp = SolutionInterp()
    tmp = sol_interp(result_tmp)
    for i in range(len(tmp)):
        print(i, tmp[i])

# import multiprocessing as mp
# vis_load_env_load = mp.Lock() # useless, should use the same locks
def visualize_kuka_traj_luo(env, traj, lock=None, is_ee3d=False, is_debug=False):
    '''
    The difference with visualize_traj is that this function directly receives 
    [a list of 1d numpy] or [2d np] as trajectory, not using a traj_agent
    
    '''
    ## the objects' trajectories, should be 2 for static obj
    max_len_traj = max([len(obj.trajectory.waypoints) for obj in env.objects])
    gifs = []
    max_len = max(len(traj), max_len_traj)
    assert max_len == len(traj)
    # print('[vis traj] max_len_traj', max_len_traj)
    print('[vis traj] max_len:', max_len)

    se_valid = check_start_end(traj) # if the start and end is valid
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
    colors = plt.cm.jet(np.linspace(0,1, max_len)) # color of the dot
    vshape_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.02)  # Red color
    has_collision = False
    num_collision_frame = 0
    collision_list = []
    add_start_end_marker(env, traj)
    assert not is_ee3d, 'depricated, prevent bugs'
    # for c, timestep in tqdm(enumerate(traj)):
    for i_t in tqdm(range(max_len)):

        if not is_ee3d: # traj is xyz-level
            env.robot.set_config(traj[i_t])
            new_pos = p.getLinkState(env.robot_id, 6)[0]
        else:
            new_pos = traj[i_t]
        cam_pos = p.getLinkState(env.robot_id, 5)[0]
        new_pos_q = p.getQuaternionFromEuler(new_pos)

        dist = 2 - np.linalg.norm(cam_pos)
        # tmp_id = p.loadURDF("sphere2red.urdf", new_pos, globalScaling=0.05, flags=p.URDF_IGNORE_COLLISION_SHAPES)
        ## slow
        tmp_id = p.createMultiBody(baseMass=0, basePosition=new_pos, baseVisualShapeIndex=vshape_id,)
        p.changeVisualShape(tmp_id, linkIndex=-1, rgbaColor=colors[i_t])

        # print(p.getBodyInfo(tmp_id))
        p.performCollisionDetection()
        c_pts = p.getContactPoints(env.robot_id)
        if c_pts is None:
            pass # also no collision
        elif len(c_pts) > 0: # very important, check is None
            has_collision = True
            num_collision_frame += 1
            collision_list.append(i_t)
            # print(f'has_collision {has_collision}; c_pts: {c_pts}')


        cam_pos = [0,0,0.3]
        dist = 1.8
        
        view_mat = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=cam_pos, distance=dist, # [0, 0, 0]
                yaw=-i_y, pitch=-30-i_p, roll=0, upAxisIndex=2) # yaw=-90-i_y
        i_p = i_p + 0.2 if i_p < 20 else i_p
        i_y += 0.3
        

        gifs.append(p.getCameraImage(width=360, height=360, lightDirection=[1, 1, 1], shadow=0,
                                                 renderer=p.ER_TINY_RENDERER,
                                                 viewMatrix=view_mat,
                                                 projectionMatrix=env.proj_mat,
                                                 flags=p.ER_NO_SEGMENTATION_MASK,
                                                 )[2])
    
    
    end_yaw = 40 + i_y
    gifs.extend([gifs[-1]]*10)
    rot_gap = 12; n_rot_frames = round(360 / rot_gap)
    for i_y in range(0, 360, rot_gap): # 2 3
        view_mat = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=cam_pos, distance=dist, # [0, 0, 0]
                yaw=(end_yaw+i_y), pitch=-40, roll=0, upAxisIndex=2)
        gifs.append(p.getCameraImage(width=360, height=360, lightDirection=[1, 1, 1], shadow=0,
                                                 renderer=p.ER_TINY_RENDERER,
                                                 viewMatrix=view_mat,
                                                 projectionMatrix=env.proj_mat,
                                                 flags=p.ER_NO_SEGMENTATION_MASK,
                                                 )[2])

    gifs = set_collision_marker(gifs, has_collision, se_valid)
    set_collision_frame(gifs, collision_list)
    print(f'[vis traj] num_collision_frame: {num_collision_frame}; start end valid: {se_valid}')
    # view_mat = p.computeViewMatrixFromYawPitchRoll(
    #             cameraTargetPosition=new_pos, distance=dist, # [0, 0, 0]
    #             yaw=0, pitch=-45, roll=0, upAxisIndex=2)
    # gifs.append(p.getCameraImage(width=360, height=360, lightDirection=[1, 1, 1], shadow=1,
    #                                              renderer=p.ER_BULLET_HARDWARE_OPENGL,
    #                                              viewMatrix=view_mat,
    #                                              projectionMatrix=env.proj_mat,
    #                                              )[2])
    ds = [100] * (len(gifs)-1)
    ds.append(6000)
    # ds[-n_rot_frames:] = [200,] * n_rot_frames # slow down, a list
    ds[0] = 1200
    vis_dict = dict(ncoll_frame=num_collision_frame, se_valid=se_valid)
    env.unload_env()
    return gifs, ds, vis_dict

NP_RED = np.array([255, 0, 0, 255], dtype=np.uint8)
NP_GREEN = np.array([0, 255, 0, 255], dtype=np.uint8)
NP_PINK = np.array([1.0, 0.078, 0.576, 0.5], dtype=np.float32) # goal large
NP_GREEN_2 = np.array([0., 1., 0., 0.5], dtype=np.float32) # start large


        

def add_start_end_marker(env, traj):
    '''add special marker at the start and end position'''
    env.robot.set_config(traj[0])
    start_pos = p.getLinkState(env.robot_id, 6)[0]
    env.robot.set_config(traj[-1])
    end_pos = p.getLinkState(env.robot_id, 6)[0]
    # pdb.set_trace()
    start_vid = p.createVisualShape(p.GEOM_SPHERE, radius=0.06)  # Red color
    tmp_id = p.createMultiBody(baseMass=0, basePosition=start_pos, baseVisualShapeIndex=start_vid,)
    p.changeVisualShape(tmp_id, linkIndex=-1, rgbaColor=NP_GREEN_2)

    end_vid = p.createVisualShape(p.GEOM_SPHERE, radius=0.06)  # Red color
    tmp_id = p.createMultiBody(baseMass=0, basePosition=end_pos, baseVisualShapeIndex=end_vid,)
    p.changeVisualShape(tmp_id, linkIndex=-1, rgbaColor=NP_PINK)

    

def set_collision_frame(gifs, collision_list):
    '''mark red exactly at every collision frame'''
    h, w, c = gifs[0].shape # (360, 360, 4), assume same resolution
    cube_h, cube_w = h // 10, w // 10
    cube_2h = cube_h * 2
    for i_cl in collision_list:
        gifs[i_cl][cube_h:cube_2h, :cube_w, :] = NP_RED

def set_collision_marker(gifs, has_collision, se_valid):
    '''set the up left corner to *Red* is detect collision, *Black* is no collision
    also set the se_valid at the up right corner
    gifs (list of np, 0-255, uint):
    '''
    h, w, c = gifs[0].shape # (360, 360, 4), assume same resolution
    cube_h, cube_w = h // 10, w // 10
    for i in range(len(gifs)):
        if has_collision:
            gifs[i][:cube_h, :cube_w, :] = NP_RED
        else:
            gifs[i][:cube_h, :cube_w, :] = NP_GREEN
        if se_valid:
            gifs[i][:cube_h, -cube_w:, :] = NP_GREEN
        else:
            gifs[i][:cube_h, -cube_w:, :] = NP_RED

    return gifs

def check_start_end(traj, thres=0.5): # 0.2
    '''check if trajectory is close to the start and end'''
    s_valid = np.linalg.norm(traj[0] - traj[1]) < thres # 0.1, do use [1] - [0]
    e_valid = np.linalg.norm(traj[-2] - traj[-1]) < thres # 0.1
    cond_1 = s_valid & e_valid
    s_valid = np.linalg.norm(traj[2] - traj[1]) < thres # 0.1, do use [1] - [0]
    e_valid = np.linalg.norm(traj[-3] - traj[-2]) < thres # 0.1
    cond_2 = s_valid & e_valid
    # print('s_valid 1', np.linalg.norm(traj[1] - traj[0]))
    # print('e_valid 2', np.linalg.norm(traj[-1] - traj[-2]))

    return cond_1 & cond_2



def from_rel2abs_path(abs_fname, rel_path):
    current_dir = osp.dirname(osp.abspath(abs_fname))
    return osp.join(current_dir, rel_path)