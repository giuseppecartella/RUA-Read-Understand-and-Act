from pyrobot import Robot
import numpy as np
import cv2 as cv2
import time
import os

from utilis_movements import Robot_Movements_Helper
from utils_signal_detection import Detection_Helper

template = cv2.imread('template.jpg', 0)

bot = Robot('locobot')
bot.camera.reset()

bot_moves = Robot_Movements_Helper(bot)
helper_detection = Detection_Helper()
found = False

while not found:

    print('Starting acquisition of the images...')
    rgb_img , d_img = bot.camera.get_rgb_depth()

    cv2.imshow('Color', rgb_img[:, :, ::-1])
    cv2.waitKey(0)

    img = cv2.cvtColor(cv2.imread(rgb_img, cv2.COLOR_BGR2RGB)) # controllare se non si spacca !!!
    found, x_c , y_c = helper_detection.look_for_signal(rgb_img, template)

    if found:
        print("Signal found!!")
        depth = helper_detection.compute_depth_distance(x_c, y_c)

        # at this point we have (x_c, y_c , depth)
        theta = np.arctan2(x_c, y_c) # correct???

        intrinsic_matrix = bot.camera.get_intrinsics() # This function returns the camera intrinsics.
        #rotation_matrix = [np.cos(theta), -np.sin(theta), ... ]

        # once we found the 3D coords, move in that direction
        if depth >= 2.0: # due metri o meno
            meters = depth/2 # dimezziamo ??
        else:       
            meters = depth - 0.20 # stop 20 cm before
        
        bot_moves.forward(meters)
    else:
        print("Signal NOT found in this direction")

        # fai fare giri e continua
        #bot_moves.left_turn() # ruota di 60 gradi
        #continue
        pass
        

