from pb_diff_envs.objects.abstract_object import AbstractObject
import pybullet as p
import numpy as np
from pb_diff_envs.objects.dynamic_object import MovableObject, DynamicObject
from pb_diff_envs.objects.trajectory import WaypointLinearTrajectory

class VoxelObject(AbstractObject):
    '''
    class to define one 3d obstale in kuka and dual kuka env
    '''
    
    def __init__(self, base_position, base_orientation, half_extents, color=None, **kwargs):
        super().__init__(**kwargs)
        self.base_position = base_position
        self.base_orientation = base_orientation

        self.half_extents = half_extents
        if color is None:
            color = np.random.uniform(0, 1, size=3).tolist() + [1]
        self.color = color


    def load2pybullet(self):
        groundColId = p.createCollisionShape(p.GEOM_BOX, halfExtents=self.half_extents)

        groundVisID = p.createVisualShape(shapeType=p.GEOM_BOX,
                                          rgbaColor=self.color,
                                          # specularColor=[0.4, .4, 0],
                                          halfExtents=self.half_extents)
        groundId = p.createMultiBody(baseMass=0,
                                     baseCollisionShapeIndex=groundColId,
                                     baseVisualShapeIndex=groundVisID,
                                     basePosition=self.base_position,
                                     baseOrientation=self.base_orientation)
        return groundId



class DynamicVoxel(VoxelObject, MovableObject, DynamicObject):
    def __init__(self, is_static=True, trajectory=None, virtual_t=2, **kwargs):
        '''
        is_static: if True, the trajectory is start position
        kwargs: should include args for VoxelObject
        '''
        if is_static:
            assert trajectory is None
            pos = kwargs['base_position']
            orn = kwargs['base_orientation']
            t = virtual_t
            pos_orn = np.concatenate((pos, orn), axis=-1)
            waypoints = np.tile(pos_orn, (t,1))
            trajectory = WaypointLinearTrajectory(waypoints=waypoints)


        super(DynamicVoxel, self).__init__(item=self, trajectory=trajectory, **kwargs)
    
    def set_config(self, config):
        # position, orientation = p.getBasePositionAndOrientation(self.item_id)
        assert len(config)==7
        p.resetBasePositionAndOrientation(self.item_id, config[:3], config[3:])
    
