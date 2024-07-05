from abc import ABC, abstractmethod
import numpy as np
import pybullet_data
import pybullet as p
import pdb
import traceback
from colorama import Fore

class AbstractEnv(ABC):

    def __init__(self, objects, robot):
        '''
        objects is a list of AbstractObject (static objects) and DynamicObject (dynamic object)
        robot is a instance of AbstractRobot
        If there are multiple robots that will be controlled simultaneously, use robot.grouping.RobotGroup to group these robots into one meta-robot first
        
        '''
        self.objects = objects
        self.robot = robot

    def load(self, **kwargs):
        self.initialize_pybullet(**kwargs)

        self.object_ids = []
        self.robot_id = None
        
        for object_ in self.objects:
            self.object_ids.append(object_.load())
        self.robot_id = self.robot.load()

        self.post_process()        
        p.performCollisionDetection()
        p.addUserDebugLine([0, 0, 0], [1, 0, 0], lineColorRGB=[1, 0, 0], lineWidth=10)
        p.addUserDebugLine([0, 0, 0], [0, 1, 0], lineColorRGB=[0, 1, 0], lineWidth=10)
        p.addUserDebugLine([0, 0, 0], [0, 0, 1], lineColorRGB=[0, 0, 1], lineWidth=10)

    def initialize_pybullet(self, reconnect=True, GUI=False, light_height_z=100):
        '''
        NOTE that in all other call p.XXX(), if no id is given, it is 0.
        the pb_id here is of no use.
        To prevent bugs, we only allow one connection.
        '''
        if reconnect:
            try:
                # close all the previous pybullet connections
                while True:
                    p.resetSimulation()
                    p.disconnect(self.pb_id)
            except Exception as e:
                if 'Not connected to physics server.' not in str(e): # print if other error
                    traceback_str = traceback.format_exc()
                    print("\033[91m" + traceback_str + "\033[0m")
        
            if GUI:
                self.pb_id = p.connect(p.GUI, options='--background_color_red=1.0 --background_color_green=1.0 --background_color_blue=1.0')
            else:
                self.pb_id = p.connect(p.DIRECT)
        
        print(Fore.GREEN + f'init self.pb_id: {self.pb_id}' + Fore.RESET)
        # assert self.pb_id == 0, 'to make sure no other connection exists'
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, lightPosition = [0, 0, light_height_z])
        if GUI:        
            self.set_camera_angle()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        p.setRealTimeSimulation(0)

    def unload_env(self):
        try:
            # close all the previous pybullet connections
            p.resetSimulation(physicsClientId=self.pb_id)
            p.disconnect(self.pb_id)
        except:
            pass

    def post_process(self):
        """
        Do nothing for parent class, optionally masking collision among groups of objects
        """           
        pass
    
    def set_camera_angle(self):
        """
        Do nothing for parent class
        """           
        pass    

    def render(self):
        """
        Return a snapshot of the current environment
        """        
        # 720
        return p.getCameraImage(width=400, height=400, lightDirection=[0, 0, -1], shadow=0, renderer=p.ER_BULLET_HARDWARE_OPENGL)[2]
    

def load_table(urdf_file):

    table_id = p.loadURDF(fileName=urdf_file)

    # Get the dimensions of the tabletop (the xy-plane wood)
    tabletop = p.getVisualShapeData(table_id)[0]
    tabletop_h = tabletop[3][2]  # Extract the Z dimension of the wood top
    frame_offset = tabletop[5][2]
    float_eps = 0.001 # prevent collision between table and other things
    final_offset = frame_offset + tabletop_h / 2 + float_eps
    p.resetBasePositionAndOrientation(table_id, posObj=(0,0,-final_offset), ornObj=(0,0,0,1))
    p.stepSimulation()
    p.getContactPoints()
    return table_id


