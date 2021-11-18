# -*- coding: utf-8 -*-
import os
import cv2
import time
#from pyrobot import Robot
import numpy as np
import matplotlib.pyplot as plt
from utils.robot_wrapper import RobotWrapper
from utils.signal_detector import SignalDetector
from utils.geometry import compute_3d_point, get_all_3d_points, get_single_3d_point, coordinate_projection
from utils import parameters
from utils.shortest_path import A_star

def quantize(planimetry, kernel, threshold_min):
    stride = kernel[0]
    
    oW = ((planimetry.shape[1] - kernel[0]) // stride) + 1
    oH = ((planimetry.shape[0] - kernel[1]) // stride) + 1

    out = np.zeros((oH, oW))

    for i in range(oH):
        for j in range(oW):
            out[i,j] = np.count_nonzero(planimetry[stride*i:stride*i+kernel[0], stride*j:stride*j+kernel[1]])
    
    out = np.where(out > threshold_min, 255, 0)
    return out.astype('float32')

def from_quantize_space_to_init(coords_point, kernel):
    x = coords_point[0] * kernel[0] + (kernel[0] // 2)
    y = coords_point[1] * kernel[1] + (kernel[1] // 2)
    return (x,y)

def from_init_to_quantized_space(coords_point, kernel):
    x = coords_point[0] // kernel[0]
    y = coords_point[1] // kernel[1]
    return (x,y)

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

def _inpaint_depth_img(depth_img):
    result = depth_img.astype(np.single)
    mask = np.where(result == 0, 255, 0).astype(np.ubyte)
    kernel = np.ones((7,7))   
    
    #result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, (7,7))
    mask = cv2.dilate(mask, kernel, iterations=2)
    result = cv2.inpaint(result, mask, 3, cv2.INPAINT_TELEA)
    result = cv2.medianBlur(result, 5)
    result = cv2.medianBlur(result, 5)
    
    #kernel = np.ones((7,7))
    #mask = cv2.erode(mask, kernel)

    return result
        
if __name__ == '__main__':
    #robot_wrapper = RobotWrapper()

    template = cv2.imread("utils/template.jpg")
    signal_detector = SignalDetector(template)
    
    #Keep moving until signal is not found. Each time performs
    ANGLES_RADIANT = np.pi/6
    MAX_ROTATIONS = 14

    found, x_c, y_c = False, None, None
    for i in range(MAX_ROTATIONS):
        print('{} Acquisition of the frame RGBD...'.format(i))
        #rgb_img, d_img = robot_wrapper.get_rgbd_frame()
        #plt.imsave('results/rgb.jpg', d_img)
        #plt.imsave('results/depth.png', d_img, cmap='gray')
        
        rgb_img = cv2.cvtColor(cv2.imread('test_images/obstacle8.png'), cv2.COLOR_BGR2RGB)
        d_img = np.load('test_images/obstacle8.npy')
        d_img = _inpaint_depth_img(d_img)
        #d_img = robot_wrapper._inpaint_depth_img(d_img)

        matrix_3d_points = np.round(get_all_3d_points(d_img) * 100).astype('int32') #transform from meters to cm
        coordinates_Z = matrix_3d_points[:,:,2]
        #mask2 = np.logical_or(coordinates_Z < (0.08 * 100), coordinates_Z > (parameters.ROBOT_HEIGHT * 100))
        
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
            plt.imsave('results/mask.png', mask, cmap='gray')
            
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

            copy_planimetry_obstacles = planimetry_obstacles
            plt.imsave('results/pure_planimetry.png', planimetry_obstacles, cmap='gray', origin='lower')
            kernel = np.ones((3,3))
            
            # serve per togliere quei puntini bianchi ?? --> da controllare in lab 
            planimetry_obstacles = cv2.medianBlur(planimetry_obstacles.astype('uint8'), 5)
            planimetry_obstacles = cv2.dilate(planimetry_obstacles, kernel, iterations=2)
            planimetry_obstacles = cv2.GaussianBlur(planimetry_obstacles, (51,51), (51-1)/5) # filtro 51 sta almeno a 20 cm da ostacoli
            plt.imsave('results/dilated_blurredplanimetry.png', planimetry_obstacles, cmap='gray', origin='lower')
            plt.matshow(planimetry_obstacles, cmap='gray', origin='lower')
            plt.show()

            QUANTIZATION_WINDOW_SIZE = (3,3)
            GAUSSIAN_FILTER_SIZE = (3,3)
            THRESHOLD = 3

            quantized_planimetry = quantize(planimetry_obstacles, QUANTIZATION_WINDOW_SIZE, THRESHOLD)
            plt.imsave('results/quantized_planimetry.png', quantized_planimetry, cmap='gray', origin='lower')
            plt.matshow(quantized_planimetry, cmap='gray', origin='lower')
            plt.show()

            #To apply gaussian filter on the quantized space
            '''quantized_planimetry = cv2.GaussianBlur(quantized_planimetry, GAUSSIAN_FILTER_SIZE, (GAUSSIAN_FILTER_SIZE[0]+0-1)/5)
            plt.matshow(quantized_planimetry, cmap='gray', origin='lower')
            plt.show()
            plt.imsave('results/quantized_planimetry_blur.png', quantized_planimetry, cmap='gray', origin='lower')'''

            #quantized_planimetry_stars = np.where(quantized_planimetry > 11, 255, 0)
            '''plt.matshow(quantized_planimetry, cmap='gray', origin='lower')
            plt.show()
            plt.imsave('results/quantized_planimetry_binary.png', quantized_planimetry, cmap='gray', origin='lower')'''

            #PATH COMPUTATION---------------------------#

            #Quantizazion phase
            middle_position = planimetry_obstacles.shape[1] // 2
            middle_position_quantized = from_init_to_quantized_space((middle_position, 0), QUANTIZATION_WINDOW_SIZE)[0]
            #print(middle_position, middle_position_quantized)
            
            #proporzione: y_vecchiosegnale:y_vecchia = ynuovasegnale:y_nuova
            y_signal = signal_3d_point[1] * planimetry_obstacles.shape[1] // old_planimetry_dimension[1]
            start = (0, middle_position)     
            end = (signal_depth - 30, middle_position - y_signal)
            #Same thing with quantized coordinates
            start_quantized = (0, middle_position_quantized)
            end_quantized = from_init_to_quantized_space(end, QUANTIZATION_WINDOW_SIZE)

            print(f"Original Space -------- Relative Robot Coordinates: {start}, Signal Coordinates: {end}")
            print(f"Quantization space ---- Relative Robot Coordinates: {start_quantized}, Signal Coordinates: {end_quantized}")           
            
            #planimetry_obstacles_aStar = np.where(planimetry_obstacles > 1.9,255, 0)
            
            a_star_alg = A_star()
            path = a_star_alg.compute(quantized_planimetry, start_quantized, end_quantized, False)
            #print(path)

            for i in path:
                coord = from_quantize_space_to_init(i, QUANTIZATION_WINDOW_SIZE)
                copy_planimetry_obstacles[coord[0], coord[1]] = 255
            plt.imsave('results/path.png', copy_planimetry_obstacles, cmap='gray', origin='lower')
            plt.matshow(copy_planimetry_obstacles, cmap='gray', origin='lower')
            plt.show()

            """pose_x, pose_y, pose_yaw = robot_wrapper.robot.base.get_state('odom')
            starting_pose = np.array([pose_x, pose_y, pose_yaw])

            old_path = (0,0)

            for i in range(30,len(path), 30):
                if old_path == (0,0):
                    y_new = path[i][1] / 100.0
                else:
                    y_new = (middle_position - path[i][1])/100.0

                print('Considered coords: {}'.format(path[i]))
                print('Old path: {}'.format(old_path))
                print('Starting pose: {}'.format(starting_pose))

                x_new = path[i][0]/100.0

                print(x_new, y_new)

                x_new -= (old_path[0] / 100.0)
                y_new -= ((middle_position - old_path[1]) / 100.0)

                print(x_new, y_new)
                current_pose = starting_pose + np.array([x_new,y_new,0.0])
                print('Current pose: {}'.format(current_pose))
                coords = coordinate_projection(starting_pose, current_pose)
                print('Final coordinates: {}'.format(coords))
                #middle_position = middle_position/100.0
                #print(middle_position - coords[0][1])
                starting_pose = current_pose
                old_path = path[i]
                
                if abs(coords[0][1]) < 0.1:
                    coords[0][1] = 0.0
                robot_wrapper.reach_relative_point(coords[0][0], coords[0][1])
                #ATTENZIONE IN OTTICA DI IMPLEMENTAZIONE DI UN WHILE 
                #CHE QUINDI PERMETTA DI NON DOVER RIAVVIARE OGNI VOLTA LO SCRIPT
                #BISOGNA RIAGGIORNARE LA GLOBAL POSITION SETTANDOLA A ZERO
                #CIO VA FATTO OGNI VOLTA CHE SI RIACQUISISCE
            """
            break
    
        #else:
            #robot_wrapper.turn(ANGLES_RADIANT)
            
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
