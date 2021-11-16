# -*- coding: utf-8 -*-
import os
import cv2
import time
from pyrobot import Robot
import numpy as np
import matplotlib.pyplot as plt
from utils import shortest_path
from utils.robot_wrapper import RobotWrapper
#from functions_obstacles import compute_3d_point, compute_different_distance, compute_paths, construct_planimetry
from utils.signal_detector import SignalDetector
from utils.geometry import compute_3d_point, get_all_3d_points, get_single_3d_point
from utils import parameters
from utils.shortest_path import A_star


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
        distances, target_position, best_distance = (compute_different_distance(bot_moves, pts, paths, depth))
        #print('Target position: {}'.format(target_position))
        if len(distances) == 0:
            print('There are some gaps but they are too small for the robot!')
            print('Exploration should start')
            pass
            return False
        
        else:
            bot_moves.reach_relative_point(target_position)
            print('Reaching next gap.')
            return True

            
if __name__ == '__main__':
    robot_wrapper = RobotWrapper()
    template = cv2.imread("utils/template.jpg")
    signal_detector = SignalDetector(template)
    
    #Keep moving until signal is not found. Each time performs
    ANGLES_RADIANT = np.pi/6
    MAX_ROTATIONS = 14

    found, x_c, y_c = False, None, None
    for i in range(MAX_ROTATIONS):
        print('{} Acquisition of the frame RGBD...'.format(i))
        rgb_img, d_img = robot_wrapper.get_rgbd_frame()
        #rgb_img = cv2.cvtColor(cv2.imread('test_images/obstacle3.png'), cv2.COLOR_BGR2RGB)
        #d_img = np.load('test_images/obstacle3.npy')
        #d_img = robot_wrapper._inpaint_depth_img(d_img)

        matrix_3d_points = np.round(get_all_3d_points(d_img) * 100).astype('int32') #transform from meters to cm
        coordinates_Z = matrix_3d_points[:,:,2]
        mask2 = np.logical_or(coordinates_Z < (0.08 * 100), coordinates_Z > (parameters.ROBOT_HEIGHT * 100))
        

        found, x_c, y_c = signal_detector.look_for_signal(rgb_img)
        signal_3d_point = np.round(get_single_3d_point(d_img, y_c, x_c)[0] * 100).astype('int32') #[X,Y,Z]
        #NB: LA Y 3D è POSITIVA A SX DEL ROBOT. 

        #map creation
        y_left = np.max(matrix_3d_points[:,:,1]) #max because y is positive to left
        y_right = np.min(matrix_3d_points[:,:,1])
        y_range = np.abs(y_left - y_right)
        

        if found:
            signal_depth = signal_3d_point[0] #get just X coordinate
            coordinates_X = matrix_3d_points[:,:,0]
            mask = np.logical_and(coordinates_Z > (0.08 * 100), coordinates_Z < (parameters.ROBOT_HEIGHT * 100))
            mask = np.logical_and(mask, coordinates_X < signal_depth)
            plt.imshow(mask, cmap='gray')
            plt.show()

            
            obstacles = matrix_3d_points[mask==True]
            x_coords = obstacles[:,0]
            middle_position = int(np.round(y_range / 2))
            y_coords = middle_position - obstacles[:,1]

            planimetry_obstacles = np.zeros((signal_depth, y_range))
            planimetry_obstacles[x_coords, y_coords] = 255
            planimetry_obstacles = np.where(planimetry_obstacles < 0, 0, planimetry_obstacles) #We put 0 for values which can become negative
            

            boundary_left = np.max(planimetry_obstacles, axis=0)
            boundary_left = np.argmax(boundary_left)

            flipped_matrix = planimetry_obstacles[:,::-1]
            boundary_right = np.max(flipped_matrix, axis=0)
            boundary_right = np.argmax(boundary_right)
            boundary_right = planimetry_obstacles.shape[1] - boundary_right
            
            old_planimetry_dimension = planimetry_obstacles.shape
            planimetry_obstacles = planimetry_obstacles[:, boundary_left:boundary_right]
            kernel = np.ones((3,3))
            planimetry_obstacles = cv2.dilate(planimetry_obstacles, kernel, iterations=1)
            planimetry_obstacles = cv2.GaussianBlur(planimetry_obstacles, (31,31), (31-1)/5)
            plt.matshow(planimetry_obstacles, cmap='gray', origin='lower')
            plt.show()
            

            #PATH COMPUTATION---------------------------#
            middle_position = planimetry_obstacles.shape[1] // 2
            #proporzione: y_vecchiosegnale:y_vecchia = ynuovasegnale:y_nuova
            y_signal = signal_3d_point[1] * planimetry_obstacles.shape[1] // old_planimetry_dimension[1]

            start = (0, middle_position)     
            end = (signal_depth, middle_position - y_signal)
            print(end)
            a_star_alg = A_star()           
            start_time = time.time()

            path = a_star_alg.compute(planimetry_obstacles, start, end, False)
            print("--- %s seconds ---" % (time.time() - start_time))
            print(path)

            for i in path:
                planimetry_obstacles[i[0], i[1]] = 255
            plt.matshow(planimetry_obstacles, cmap='gray', origin='lower')
            plt.show()

            
            for i in range(15,len(path), 15):
                x = i[0]
                y = i[1]
                robot_wrapper.reach_absolute_point(x,y)
                #ATTENZIONE IN OTTICA DI IMPLEMENTAZIONE DI UN WHILE 
                #CHE QUINDI PERMETTA DI NON DOVER RIAVVIARE OGNI VOLTA LO SCRIPT
                #BISOGNA RIAGGIORNARE LA GLOBAL POSITION SETTANDOLA A ZERO
                #CIO VA FATTO OGNI VOLTA CHE SI RIACQUISISCE
            break
    
        else:
            robot_wrapper.turn(ANGLES_RADIANT)
            
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