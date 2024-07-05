from .sequence import *
from diffuser.datasets.data_api import load_environment
# from diffuser.datasets.keypt_dataset import KeyPtDataset

import numpy as np
## store data constants for normalization

KUKA7D_MIN = np.array([
                -2.96705972839,
                -2.09439510239, 
                -2.96705972839, 
                -2.09439510239,
                -2.96705972839,
                -2.09439510239,
                -3.05432619099], dtype=np.float32)
KUKA7D_MAX = np.array([
                2.96705972839,
                2.09439510239,
                2.96705972839,
                2.09439510239,
                2.96705972839,
                2.09439510239,
                3.05432619099], dtype=np.float32)

DUALKUKA14D_MIN = np.array([
                -2.96705972839,
                -2.09439510239, 
                -2.96705972839, 
                -2.09439510239,
                -2.96705972839,
                -2.09439510239,
                -3.05432619099,
                ## 2
                -2.96705972839,
                -2.09439510239, 
                -2.96705972839, 
                -2.09439510239,
                -2.96705972839,
                -2.09439510239,
                -3.05432619099], dtype=np.float32)

DUALKUKA14D_MAX = np.array([
                2.96705972839,
                2.09439510239,
                2.96705972839,
                2.09439510239,
                2.96705972839,
                2.09439510239,
                3.05432619099,
                # 2
                2.96705972839,
                2.09439510239,
                2.96705972839,
                2.09439510239,
                2.96705972839,
                2.09439510239,
                3.05432619099], dtype=np.float32)

RAND_MAZE_MIN = np.array([0., 0.], dtype=np.float32)

RAND_MAZESIZE_55 = np.array([5., 5.], dtype=np.float32)
