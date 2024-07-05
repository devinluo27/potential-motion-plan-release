import matplotlib.pyplot as plt
import random
from typing import List
import numpy as np
from tqdm import tqdm
import os
from pb_diff_envs.utils.kuka_utils_luo import from_rel2abs_path
import pdb, pickle
from pb_diff_envs.environment.rand_rec_group import RandRectangleGroup, is_recWall_overlap, RectangleWall

class RandRectangleGroup_43(RandRectangleGroup):
    def __init__(self, env_list_name:str, num_groups, num_walls, 
                 maze_size, half_extents, gap_betw_wall, seed, gen_data,
                 robot_class, is_eval=False,
                 **kwargs) -> None:

        self.two = 2
        self.mode_list = []
        super().__init__(
                 env_list_name, num_groups, num_walls, 
                 maze_size, half_extents, gap_betw_wall, seed, gen_data,
                 robot_class, is_eval, **kwargs)
        

    def create_envs(self):
        '''set rec_loc_grp_list'''
        self.mode_list = []
        rec_loc_grp_list: list[np.ndarray] = [] # a list of np2d: n_c,3
        rec_hExt_grp_list = []
        for _ in tqdm(range(self.num_groups)):
            recloc_grp, rec_hExt_grp, env_modes = self.sample_one_valid_env()
            rec_loc_grp_list.append(recloc_grp) # center location, np2d
            rec_hExt_grp_list.append(rec_hExt_grp) # np2d
            self.mode_list.append(env_modes)
        
        # np3d: ng, n_cube, 2
        self.rec_loc_grp_list = np.array(rec_loc_grp_list)
        self.rec_hExt_grp_list = np.array(rec_hExt_grp_list)

        self.save_rlg_list()

        return
    
    def sample_43square(self):
        center_pos, hExt = self.sample_xy_hExt()
        # low (inclusive) to high (exclusive)
        mode = self.rng.integers(low=0, high=4)
        ## mode hint: remove the small square in the below index: 
        # 1 2
        # 4 3
        center_list = [
            [center_pos[0] - hExt[0], center_pos[1] + hExt[1] ],
            [center_pos[0] + hExt[0] , center_pos[1] + hExt[1] ],
            [center_pos[0] + hExt[0] , center_pos[1] - hExt[1] ],
            [center_pos[0] - hExt[0] , center_pos[1] - hExt[1] ],
        ]


        del center_list[mode]
        center_list = np.array( center_list )
        return center_list, hExt, mode 
    
    def check_center_list43(self, center_list, hExt, rec_list):
        ## return True if there is overlap
        rec_list = [_ for _ in rec_list] # tricks, create a new list
        tmp_recs = []
        # print(center_list)
        for center_pos in center_list:
            tmp_rec = RectangleWall(center_pos, hExt)
            has_overlap = self.check_is_overlap43(rec_list, tmp_rec, self.gap_betw_wall)
            if has_overlap:
                return True, None
            else:
                # rec_list.append(tmp_rec)
                tmp_recs.append(tmp_rec)
        
        return False, tmp_recs



    def sample_one_valid_env(self):
        '''create one env config without any overlap
        returns:
        array (n_c, 3) of 3d position '''

        rec_list = []
        env_modes = []
        cnt_iter = 0
        while len(rec_list) < self.num_walls: # * self.three:
            cnt_iter += 1
            if cnt_iter > self.rand_iter_limit: # deadlock, reset everything
                print('reach limit reset')
                for rec in rec_list: 
                    print(rec.center)
                rec_list = []
                env_modes = []
                cnt_iter = 0
                

            center_list43, half_hExt, mode = self.sample_43square()
            has_overlap, tmp_recs = self.check_center_list43(center_list43, half_hExt, rec_list)

            if has_overlap:
                pass
            else:
                # pos_list.append(center_pos.tolist()) # for sorting
                # hExt_list.append(hExt)
                rec_list.extend( tmp_recs )
                env_modes.append(mode)
                # print('rec_list', len(rec_list), tmp_recs)


        # pre sorted
        pos_list, hExt_list = self.get_pos_hExt_list(rec_list)
        

        pos_list = np.array(pos_list) # n_c, 2
        hExt_list = np.array(hExt_list)
        # print('2 pos_list', pos_list)
        return pos_list, hExt_list, env_modes
    


    def check_is_overlap43(self, rec_list, rec_new, gap_betw_wall):
        for rec in rec_list:
            is_ovlp = is_recWall_overlap(rec, rec_new, gap_betw_wall)
            if is_ovlp: # overlap
                return True
        return False



    # ------------- helpers for sampling ----------------
    def sample_xy_hExt(self):
        '''simply sample a location in all valid range, not consider overlap yet'''
        hExt = self.half_extents.copy()
        bottom_left, top_right = self.get_rec_center_range(hExt * self.two)

        center_pos = self.rng.uniform(low=bottom_left, high=top_right) # (2,)
            

        return center_pos, hExt


    def get_pklname_43(self):
        self.prefix = from_rel2abs_path(__file__, '../datasets/rand_rec2dgrp/')
        os.makedirs(self.prefix, exist_ok=True)
        if self.is_eval:
            npyname = f'{self.prefix}/{self.env_list_name}_43_eval.pkl'
        else:
            npyname = f'{self.prefix}/{self.env_list_name}_43.pkl'
        return npyname
    
    def save_center_and_mode(self, is_save=True):
        '''
        each concave obstacle consists of 3 smaller blocks,
        this function save the center and mode of each concave obstacle
        mode: the layout of the 3 smaller blocks (in {1,2,3,4}), e.g.,

        '''
        self.centers_43 = []
        for wlocs in self.rec_loc_grp_list:
            env_c = []
            # print('wlocs:', wlocs.shape) # 21, 2
            for i in range(0, len(wlocs), 3):
                x_max = max(  wlocs[i][0], wlocs[i+1][0]  ) # x max
                x_min = min(  wlocs[i][0], wlocs[i+2][0] )
                x = ( x_min + x_max ) / 2
                y = ( wlocs[i][1]  + wlocs[i+2][1] ) / 2
                env_c.append( [x, y]  )

            self.centers_43.append(env_c)
        
        self.centers_43 = np.array(self.centers_43)
        ## to avoid mode = 0
        self.mode_list = self.compute_mode_list() + 1 # choices: [1, 2, 3, 4]
        if is_save:
            pklname = self.get_pklname_43()
            d_tmp = {'centers_43': self.centers_43, 'mode_list': self.mode_list}
            if os.path.exists(pklname):
                with open(pklname, 'rb') as file:
                    read_pkl = pickle.load(file)
                assert np.isclose(read_pkl['centers_43'], self.centers_43).all()
            else:
                with open(pklname, 'wb') as file:
                    pickle.dump(d_tmp, file)
        
        return np.concatenate( [self.centers_43, self.mode_list[..., None]], axis=2 )
    
    def compute_mode_list(self):
        self.mode_list = []
        for wlocs in self.rec_loc_grp_list:
            env_m = []
            # print('wlocs:', wlocs.shape) # 21, 2
            for i in range(0, len(wlocs), 3):
                if wlocs[i][1] == wlocs[i+1][1]:
                    if wlocs[i][0] == wlocs[i+2][0]:
                        mode = 2
                    else:
                        assert wlocs[i+1][0] == wlocs[i+2][0]
                        mode = 3
                else:
                    if wlocs[i][0] == wlocs[i+2][0]:
                        mode = 1
                    else:
                        assert wlocs[i][0] == wlocs[i+1][0]
                        mode = 0
                env_m.append( mode  )

            self.mode_list.append(env_m)
        return np.array( self.mode_list )
        

