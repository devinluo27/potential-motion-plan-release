from abc import ABC, abstractmethod
from pb_diff_envs.objects.abstract_object import AbstractObject
from pb_diff_envs.objects.trajectory import AbstractTrajectory
import pybullet as p


class MovableObject(AbstractObject, ABC):
    '''all robots are set to Movable'''
    
    @abstractmethod
    def set_config(self, config):
        raise NotImplementedError
        

class MovableBaseObject(MovableObject):

    def __init__(self, move_mode, **kwargs):
        super(MovableBaseObject, self).__init__(**kwargs)
        assert (move_mode=='p') or (move_mode=='o') or (move_mode=='po') or (move_mode=='op')
        self.move_mode = move_mode
    
    def set_config(self, config):
        '''
        static object does not have this,
        enable this interface for static obj
        '''
        position, orientation = p.getBasePositionAndOrientation(self.item_id)
        # print('config', config) # (7,) fro voxel? pos+orn
        if self.move_mode == 'p':
            assert len(config)==3
            p.resetBasePositionAndOrientation(self.item_id, config, orientation)
        elif self.move_mode == 'o':
            assert len(config)==4
            p.resetBasePositionAndOrientation(self.item_id, position, config)
        else:
            assert len(config)==7
            p.resetBasePositionAndOrientation(self.item_id, config[:3], config[3:])


class DynamicObject(AbstractObject):
    '''
    put movable object into dynamic object
    we update the self.item at a given timestep
    all wrapper for movable objects'''

    def __init__(self, item: MovableObject, trajectory: AbstractTrajectory, **kwargs):
        self.item = item
        self.trajectory = trajectory
        super(DynamicObject, self).__init__(**kwargs)
        # print('DynamicObject') # 1

    def set_config_at_time(self, t):
        ## maybe linear interp, depends on traj class type 
        spec = self.trajectory.get_spec(t)
        # set the spec of time t to item
        self.trajectory.set_spec(self.item, spec)

    def load2pybullet(self, **kwargs):
        print('loading dynamic object')
        item_id = self.item.load2pybullet(**kwargs)
        return item_id
        

class MovableObjectFactory:      
    @staticmethod
    def create_movable_object_class(ObjectX, MovableXObject):
        '''
        Argument: turning an object class into a movable object class
        ObjectX: class
        MovableObject: class
        '''
        assert not issubclass(ObjectX, MovableObject)
        assert issubclass(MovableXObject, MovableObject)
        # what is that?
        class MovableSpecificObject(ObjectX, MovableXObject):
            '''just a pure inheritance'''
        # print('MovableObjectFactory')
        return MovableSpecificObject


class DynamicObjectFactory:

    @staticmethod
    def create_dynamic_object_class(MovableXObject):
        '''
        Argument: turning a movable object class into a dynamic object class
        MovableXobject: default to be MovableObjectFactory
        kwargs: play the tricks, can pass all args to subclasses
        1. VoxelObject; 2. MovableBaseObject
        move_mode to 
        '''
        assert issubclass(MovableXObject, MovableObject)
        class DynamicSpecificObject(MovableXObject, DynamicObject):
            def __init__(self, trajectory: AbstractTrajectory, **kwargs):
                super(DynamicSpecificObject, self).__init__(item=self, trajectory=trajectory, **kwargs)
        return DynamicSpecificObject

