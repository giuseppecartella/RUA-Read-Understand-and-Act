import numpy as np

BASE_ROBOT = 0.35
BASE_HEIGHT = 0.63
OBSTACLE_LIMIT_DISTANCE = 0.50 # se trova qualcosa entro 0.5, si ferma perchè è ostacolo
ANGLES_RADIANT = np.pi/6 # 30 gradi
MAX_ROTATIONS = 13
STOP_DISTANCE_LIMIT = 0.3
INTERMEDIATE_STEP = 1.0 