from pb_diff_envs.environment.rand_cuboid_group import RandCuboidGroup
from typing import List
import pdb
import numpy as np

class ComposedRandCuboidGrp:
    def __init__(self, env_list_name: str, num_groups, n_comp,
                 num_walls_c, start_end, void_range,
                 half_extents_c, seed, gen_data, 
                 robot_class, is_eval=False, **kwargs) -> None:
        
        assert num_walls_c.ndim == 1 and num_walls_c.shape[0] >= 2
        assert half_extents_c.ndim == 2 and half_extents_c.shape[1] == 3
        
        self.num_groups = num_groups
        self.n_comp = n_comp
        self.grp_list: List[RandCuboidGroup] = []
        self.base_orn = [0, 0, 0, 1]

        for i_c in range(n_comp):
            el_name = f'{env_list_name}_{i_c}'
            seed_tmp = seed + i_c if kwargs['grp_seed_diff'] else seed
            tmp = RandCuboidGroup(
                el_name, num_groups, num_walls_c[i_c],
                start_end, void_range, half_extents_c[i_c], seed_tmp, gen_data,
                robot_class, is_eval, **kwargs,
                
            ) 
            self.grp_list.append(tmp)
        
        self.create_envs()

    def create_envs(self):
        '''
        compose multiple rand rec group, and put into a list
        '''
        ## can use np split to speed up
        self.loc_grp_list = [] # a list of [     ]
        self.hExt_grp_list = []
        for i_g in range(self.num_groups):
            wloc_c = [] # list [ np[n_w, 2], ..., np[n_w, 2] ],
            hExt_c = []
            for i_c in range(self.n_comp):
                wloc_c.append( self.grp_list[i_c].voxel_grp_list[i_g] )
                hExt_c.append( self.grp_list[i_c].voxel_hExt_grp_list[i_g] )

            self.loc_grp_list.append( wloc_c )
            self.hExt_grp_list.append( hExt_c )
        
        tmp1 = []
        tmp2 = []
        for i_c in range(self.n_comp):
            tmp1.append(self.grp_list[i_c].voxel_grp_list) # (ng, nw, d)
            tmp2.append(self.grp_list[i_c].voxel_hExt_grp_list)
        
        self.wallLoc_list = np.concatenate(tmp1, axis=1) # (ng, nw1+nw2, d)
        self.hExt_list = np.concatenate(tmp2, axis=1)


    def get_composed_item(self, idx):
        '''
        return 1 comp env
        a list, len n_c, each elem is np [n_w, dim]
        '''
        
        wloc_c = self.loc_grp_list[idx]
        hExt_c = self.hExt_grp_list[idx]

        return wloc_c, hExt_c
    
    
