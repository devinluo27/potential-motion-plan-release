import numpy as np


class ObstaclePositionWrapper():
    '''
    return representation of obstacles as concatenated vector of positions
    '''
    def __init__(self, baseObject):
        self.__class__ = type(baseObject.__class__.__name__,
                              (self.__class__, baseObject.__class__),
                              {})
        self.__dict__ = baseObject.__dict__

    def get_obstacles(self):
        return np.array([list(obj.base_position)+list(obj.half_extents) for obj in self.objects])