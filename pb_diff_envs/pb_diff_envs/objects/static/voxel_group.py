import pybullet as p
import numpy as np
from pb_diff_envs.objects.static.voxel import DynamicVoxel
from typing import Union


class DynVoxelGroup:
    def __init__(self, xyz_list, orn_list, hExt_list, colors, is_static) -> None:
        self.num_voxels = len(xyz_list)
        self.xyz_list = xyz_list
        self.orn_list = orn_list
        self.hExt_list = hExt_list
        self.dv_list: list[DynamicVoxel] = [] # actual input to robot_env
        self.colors = colors
        self.is_static = is_static
        for i_v in range(self.num_voxels):
            pos = xyz_list[i_v]
            # print(pos, self.orn_list[i_v])
            dv = DynamicVoxel(
                base_position=pos,
                base_orientation=self.orn_list[i_v],
                half_extents=self.hExt_list[i_v],
                color=colors[i_v],
                is_static=is_static[i_v] if type(is_static) == list else is_static,
                )
            self.dv_list.append(dv)




