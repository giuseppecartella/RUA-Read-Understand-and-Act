# -*- coding: utf-8 -*-

import numpy as np

BASE_ROBOT = 0.35
ROBOT_HEIGHT = 0.63
FLOOR_HEIGHT_LIMIT = 0.08
OBSTACLE_LIMIT_DISTANCE = 0.50 # se trova qualcosa entro 0.5, si ferma perchè è ostacolo
ANGLES_RADIANT = np.pi/6 # 30 gradi
MAX_ROTATIONS = 13
STOP_DISTANCE_LIMIT = 0.3
INTERMEDIATE_STEP = 1.0
DEPTH_MAP_FACTOR = 1.0
QUANTIZATION_WINDOW_SIZE = (3,3)
GAUSSIAN_FILTER_SIZE = (3,3)
THRESHOLD = 1

trans = [0.04474711619328203, 0.026835823604535163, 0.5771852573495844]

rot = np.array([[-0.00462177,  0.00221659,  0.99998686],
                [-0.99993457, -0.01047387, -0.00459832],
                [ 0.01046354, -0.99994269,  0.00226486]])

T = np.array([[-0.00462177,  0.00221659,  0.99998686,  0.04474712],
              [-0.99993457, -0.01047387, -0.00459832,  0.02683582],
              [ 0.01046354, -0.99994269,  0.00226486,  0.57718526],
              [ 0.        ,  0.        ,  0.        ,  1.        ]])

intrinsic_matrix = np.array([[613.19714355, 0.0         , 314.70608521],
                             [0.0         , 613.91461182, 237.9289093 ],
                             [0.0         , 0.0         , 1.0         ]])