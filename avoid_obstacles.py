# -*- coding: utf-8 -*-
import argparse
import pyrobot
import ros
import cv2
import numpy as np
from utils import project_parameters as params
from utils.robot_wrapper import RobotWrapper
from utils.signal_detector import SignalDetector
from utils.img_processing import ImgProcessing
from utils.plotter import Plotter
from utils.geometry_transformation import GeometryTransformation
from utils.map_constructor import MapConstructor
from utils.path_planner import PathPlanner
import copy
import matplotlib.pyplot as plt
import torch
from PIL import Image
from vision_transformer.cam import Cam

def main():
    #----------------------INITIALIZATION-------------------------------------#
    parser = argparse.ArgumentParser()
    parser.add_argument('-lmd', '--lab_mode', type=str,
                        help='Specify if you are launching code in laboratory', required=True)
    parser.add_argument('-dbg', '--debug', type=str,
                        help='If you want to save plots', required=True)

    args = parser.parse_args()
    lab_mode = args.lab_mode
    debug = args.debug

    if lab_mode not in ['True', 'False'] or debug not in ['True', 'False']:
        print('Please specify the correct command arguments!')
        return -1

    robot_wrapper = RobotWrapper(lab_mode)
    signal_detector = SignalDetector(lab_mode=lab_mode)
    map_constructor = MapConstructor()
    path_planner = PathPlanner()
    img_processing = ImgProcessing()
    cam = Cam()
    gt = GeometryTransformation()
    plotter = Plotter('results')
    #--------------------------------------------------------------------------#

    signal_abs_coords = None

    while True:
        print('entrato nel whileee')
        #-------------------------------SIGNAL DETECTION---------------------------#
        if lab_mode == "True":
            robot_wrapper.reset_camera()
            rgb_img, d_img = robot_wrapper.get_rgbd_frame()
        else:
            rgb_img = cv2.cvtColor(cv2.imread('test_images/obstacle9.png'), cv2.COLOR_BGR2RGB)
            d_img = np.load('test_images/obstacle9.npy')

        if debug == "True":
            plotter.save_image(rgb_img, 'rgb_image', False)
            plotter.save_image(d_img, 'depth_image', True)

        d_img = img_processing.inpaint_depth_img(d_img)

    
        found_written_sign, prediction = signal_detector.look_for_written_signal(rgb_img, d_img, robot_wrapper)

        if found_written_sign:
            step_forward = robot_wrapper.get_values_for_action(prediction)
            signal_3d_point = None
            signal = False
        else :           
            signal = True
            found, x_c, y_c = signal_detector.look_for_signal(rgb_img) #y_c is the row idx, x_c is col_idx
            
            if not found and signal_abs_coords is not None:
                angle_movement = robot_wrapper.allineate_robot(signal_abs_coords)
                robot_wrapper.turn(angle_movement)
                continue
                        
            #magari sostituire queste due righe con la continue.
            #rgb_img, d_img = robot_wrapper.get_rgbd_frame()              
            #found, x_c, y_c = signal_detector.look_for_signal(rgb_img)

            
            if not found:
                rgb_img, d_img, x_c , y_c, step_forward  = robot_wrapper.explore(signal_detector, map_constructor, img_processing, gt, path_planner, signal_abs_coords)
                if step_forward is not None:
                    step_forward = robot_wrapper.get_values_for_action(prediction)
                    signal_3d_point = None
                    signal = False
                else:
                    if x_c is None:
                        rgb_img, d_img, x_c , y_c, step_forward = robot_wrapper.explore(signal_detector, map_constructor, img_processing, gt, path_planner, signal_abs_coords, last=True)
                        if step_forward is not None:
                            step_forward = robot_wrapper.get_values_for_action(prediction)
                            signal_3d_point = None
                            signal = False
                            
                        if x_c is None:
                            return -1
            else:  

                #--------------------------------------------------------------------------#

                #------------------------GEOMETRIC TRANSFORMATIONS-------------------------#           

                signal_3d_point = gt.get_single_3d_point(d_img, y_c, x_c)

                x_signal = signal_3d_point[0]
                y_signal = signal_3d_point[1]
                signal_distance = np.sqrt((x_signal - 0)**2 + (y_signal - 0)**2) #consider relative distance from robot
                signal_distance = round(signal_distance, 2)
                if signal_abs_coords is None:
                    if lab_mode == "True":
                        signal_abs_coords = gt.update_signal_abs_coords(signal_3d_point, robot_wrapper)

              
                if signal_distance /  100 <= 0.50:
                    if signal_abs_coords is not None:
                        angle_movement = robot_wrapper.allineate_robot(signal_abs_coords)
                        robot_wrapper.turn(angle_movement)
                    break
        

        matrix_3d_points = gt.get_all_3d_points(d_img)
        #--------------------------------------------------------------------------#

        #--------------------------MAP CONSTRUCTION--------------------------------#
        #For Locobot Y is positive to left.
        planimetry, robot_coords, signal_coords = map_constructor.construct_planimetry(matrix_3d_points, signal_3d_point, signal = signal)
        if debug == "True":
            copy_planimetry = copy.deepcopy(planimetry)
            plt.imsave('results/raw_planimetry.png', copy_planimetry, cmap='gray', origin='lower')

        print('signal coords sono: ', signal_coords)
        planimetry = img_processing.process_planimetry(planimetry, signal_coords)

        if debug == "True":
            plotter.save_image(rgb_img, 'rgb_image', False)
            plotter.save_image(d_img, 'depth_image', True)

            copy_planimetry = copy.deepcopy(planimetry)
            plt.imsave('results/processed_planimetry.png', copy_planimetry, cmap='gray', origin='lower')


        #-----------------------------PATH DEFINITION------------------------------#
        start_point = robot_coords
        step_forward=150

        if signal_coords is not None:
            end_point_signal = (signal_coords[0], signal_coords[1])
            end_point_path = (signal_coords[0] - 15, signal_coords[1])
        else:
            end_point_signal = (step_forward, robot_coords[1])
            end_point_path = (step_forward - 15, robot_coords[1])
        
        path = path_planner.compute(planimetry, start_point, end_point_path, False)
        
        if path is not None:
            if debug == "True":
                plotter.save_planimetry(planimetry, start_point, end_point_signal, 'planimetry_with_trajectory', coords=path)
                
            path = path_planner.shrink_path(path)
            path = [(0, 46), (53, 55), (56, 63), (59, 70), (62, 74), (65, 77), (69, 79), (78, 83), (84, 84), (93, 85), (103, 85), (145, 85)]
            if len(path) >= 2:
                path = path_planner.clean_shrink_path(path, end_point_path)
   
            if debug == "True":
                plotter.save_planimetry(planimetry, start_point, end_point_signal, 'planimetry_with_shrinked_path', coords=path)

            #--------------------------------------------------------------------------#

            #-----------------------------FOLLOW TRAJECTORY---------------------------#
            
            if lab_mode == "True":
                robot_wrapper.follow_trajectory(path, robot_coords)
                     
            if signal_abs_coords is not None:
                robot_pose = robot_wrapper.get_robot_position()
                distance = np.sqrt( (robot_pose[0] - signal_abs_coords[0]) ** 2 + (robot_pose[1] - signal_abs_coords[1]) ** 2)
                distance = round(distance, 2)

                if distance <= 0.50:
                    if signal_abs_coords is not None:

                        angle_movement = robot_wrapper.allineate_robot(signal_abs_coords)
                        robot_wrapper.turn(angle_movement)
                    break
        else:
            break

    
    print('Arrived to destination!')

if __name__ == '__main__':
    main()