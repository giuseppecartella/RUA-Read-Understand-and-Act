# -*- coding: utf-8 -*-
import os
import cv2
from pyrobot import Robot
import numpy as np
from utils.robot_movements import Robot_Movements_Helper
from utils.signal_detection import Detection_Helper
from functions_obstacles import compute_3d_point, compute_different_distance, compute_paths, construct_planimetry



def reach_signal(bot_moves, helper_detection, poi, d_img):
    #NB: STEP PRELIMINARI DA FARE: scommentare la giusta 3d point
    #dopo aver verificato che i singoli casi funzionano implementare il while.

    INTERMEDIATE_STEP = 0.5
    x_c, y_c = int(round(poi[0])), int(round(poi[1]))

    depth = helper_detection.inpaint_d_img(d_img)
    pts = compute_3d_point(bot_moves.robot, [x_c, y_c], depth)
    planimetry = construct_planimetry(depth) #solo della prima fascia

    if not np.any(planimetry[0]):
        print('No obstacles at all. I can go everywhere until the step (0.5 cm for example)')
        bot_moves.reach_relative_point([INTERMEDIATE_STEP, pts[0][1], 0.0])
        print('Step done. Process is stopping!')
        return True

    paths = compute_paths(planimetry[0])

    if len(paths) == 0: 
        print('No path found. I should explore the neighborhood!')
        return False
    else:
        distances, target_position, best_distance = compute_different_distance(bot_moves, pts, paths, depth)
        if len(distances) == 0:
            print('There are some gaps but they are too small for the robot!')
            print('Exploration should start')
            return False
        
        else:
            bot_moves.reach_relative_point(target_position)
            print('Reaching next gap.')
            return True

            
if __name__ == '__main__':
    bot = Robot('locobot')
    bot.camera.reset()

    bot_moves = Robot_Movements_Helper(bot)
    template = cv2.imread("utils/template.jpg")
    helper_detection = Detection_Helper(template)
    
    #Keep moving until signal is not found. Each time performs
    ANGLES_RADIANT = np.pi/6
    MAX_ROTATIONS = 13

    found, x_c, y_c = False, None, None
    for i in range(MAX_ROTATIONS):
        print('{} Acquisition of the frame RGBD...'.format(i))
        rgb_img, d_img = bot_moves.read_frame()
        #rgb_img = cv2.cvtColor(cv2.imread('prova.png'), cv2.COLOR_BGR2RGB)
        #d_img = np.load('prova.npy')

        found, x_c, y_c = helper_detection.look_for_signal(rgb_img)
        if found:
            break
        else:
            bot_moves.left_turn(ANGLES_RADIANT)

    if found:
        #signal is found, so now we can manage the robot movement.
        print('Signal found...reaching it!')
        is_arrived = reach_signal(bot_moves, helper_detection, [x_c, y_c], d_img)
        
        if is_arrived:
            print('Robot arrived to destination...In front of the signal!')
        else:
            print('Something went wrong! Robot no longer sees the signal!')
    else:
        print('Stop signal NOT FOUND in the neighborhood... Hence, robot will not move!')