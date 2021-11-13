# -*- coding: utf-8 -*-
import os
import cv2
#from pyrobot import Robot
import numpy as np
import matplotlib.pyplot as plt
from utils.robot_wrapper import RobotWrapper
#from functions_obstacles import compute_3d_point, compute_different_distance, compute_paths, construct_planimetry
from utils.signal_detector import SignalDetector
from utils.geometry import compute_3d_point, get_all_3d_points, get_single_3d_point
from utils import parameters


def reach_signal(bot_moves, helper_detection, poi, d_img):
    #NB: STEP PRELIMINARI DA FARE: scommentare la giusta 3d point
    #dopo aver verificato che i singoli casi funzionano implementare il while.

    INTERMEDIATE_STEP = 0.5
    x_c, y_c = int(round(poi[0])), int(round(poi[1]))

    depth = helper_detection.inpaint_d_img(d_img)
    masked_depth = np.where(depth <= 0.5, 255, 0)
    cv2.imwrite('Depthimage.jpg', masked_depth)
    pts = compute_3d_point(bot_moves.robot, [x_c, y_c], depth)
    print(pts)
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
        print('Target position: {}'.format(target_position))
        if len(distances) == 0:
            print('There are some gaps but they are too small for the robot!')
            print('Exploration should start')
            return False
        
        else:
            bot_moves.reach_relative_point(target_position)
            print('Reaching next gap.')
            return True

            
if __name__ == '__main__':
    #bot = Robot('locobot')
    #bot.camera.reset()

    #bot_moves = Robot_Movements_Helper(bot)
    robot_wrapper = RobotWrapper()
    template = cv2.imread("utils/template.jpg")
    signal_detector = SignalDetector(template)
    
    #Keep moving until signal is not found. Each time performs
    ANGLES_RADIANT = np.pi/6
    MAX_ROTATIONS = 13

    found, x_c, y_c = False, None, None
    for i in range(MAX_ROTATIONS):
        print('{} Acquisition of the frame RGBD...'.format(i))
        #rgb_img, d_img = bot_moves.read_frame()
        rgb_img = cv2.cvtColor(cv2.imread('obstacle3.png'), cv2.COLOR_BGR2RGB)
        d_img = np.load('obstacle3.npy')

        d_img = robot_wrapper._inpaint_depth_img(d_img)
        #plt.imshow(d_img, cmap='gray')
        #plt.show()

        matrix_3d_points = np.round(get_all_3d_points(d_img) * 100).astype('int32') #transform from meters to cm
        mask = np.logical_or(matrix_3d_points[:,:,2] < (0.1 * 100), matrix_3d_points[:,:,2] > (parameters.ROBOT_HEIGHT * 100))
        #var = np.where(mask == False, d_img, 0)
        #100 because we transform from meters to cm
      
        found, x_c, y_c = signal_detector.look_for_signal(rgb_img)
        signal_3d_point = np.round(get_single_3d_point(d_img, y_c, x_c)[0] * 100).astype('int32') #[X,Y,Z]
        #NB: LA Y 3D è POSITIVA A SX DEL ROBOT. 

        #map creation
        y_left = np.max(matrix_3d_points[:,:,1])
        y_right = np.min(matrix_3d_points[:,:,1])
        y_range = np.abs(y_left - y_right)
    
        if found:
            signal_depth = signal_3d_point[0] #get just X coordinate

            #var = np.where(var < signal_depth/100, var, 0)
            #plt.imshow(var, cmap='gray')
            #plt.show()
            planimetry_obstacles, planimetry_free = np.zeros((signal_depth, y_range)), np.zeros((signal_depth, y_range))
            #plt.matshow(planimetry, cmap='gray', origin='lower')
            #plt.show()

            #obstacles = np.logical_and(matrix_3d_points[mask == False], matrix_3d_points[:,:,0] < signal_depth)
            obstacles = matrix_3d_points[[mask == False] and [matrix_3d_points[:,:,0] < signal_depth]] #consider only points that are obstacles
            free_points = matrix_3d_points[matrix_3d_points[:,:,2] < 0.1 * 100]
            free_points = free_points[free_points[:, 0] < signal_depth]
            

            middle_position = int(np.round(y_range / 2))
            free_points = free_points[:,[0,1]]
            obstacles = obstacles[:,[0,1]]
      
            x_planimetry_free = free_points[:,0]
            y_planimetry_free = middle_position - (free_points[:,1])

            x_planimetry_obst = obstacles[:,0]
            y_planimetry_obst = middle_position - (obstacles[:,1])

            planimetry_free[x_planimetry_free, y_planimetry_free] = 255
            planimetry_obstacles[x_planimetry_obst, y_planimetry_obst] = 255

            final_planimetry = planimetry_obstacles - planimetry_free
            kernel = np.ones((3,3))
            final_planimetry = cv2.dilate(final_planimetry, kernel, iterations=2)
            plt.matshow(final_planimetry, cmap='gray', origin='lower')
            plt.show()

            break

            #now we have to insert ones that indicates obstacles

            
        else:
            #bot_moves.left_turn(ANGLES_RADIANT)
            pass


    
    """
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
    """