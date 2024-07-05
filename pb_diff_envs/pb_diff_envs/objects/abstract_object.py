from abc import ABC, abstractmethod


class AbstractObject(ABC):
    ''' generic class
    could be a robot or a movable/static object
    '''
    def __init__(self, **kwargs):
        super(AbstractObject, self).__init__()

    def load(self, **kwargs):
        result = self.load2pybullet(**kwargs)
        if isinstance(result, list):
            self.item_ids = result
        elif isinstance(result, int):
            self.item_id = result
        else:
            assert False
        return result

    @abstractmethod
    def load2pybullet(self, **kwargs):
        '''
        load into PyBullet and return the id of robot
        '''        
        raise NotImplementedError