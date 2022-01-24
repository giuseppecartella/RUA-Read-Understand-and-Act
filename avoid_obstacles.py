import argparse
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
import matplotlib.pyplot as plt
import time

from vision_transformer.predict import SignsReader

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
    gt = GeometryTransformation()
    signs_reader = SignsReader()
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


        found, x_c, y_c = signal_detector.look_for_signal(rgb_img) #y_c is the row idx, x_c is col_idx
        
        if not found:
            print('Signal not found. STARTING EXPLORATION.')
            print(signal_abs_coords)
            # !!!!!  DA SCOMMENTARE SHRINK E GO_TO_RELATIVE DENTRO A EXPLORE() !!!!!!
            rgb_img, d_img, x_c , y_c = robot_wrapper.explore(signal_detector, map_constructor, img_processing, gt, path_planner, signal_abs_coords)
            if x_c is None:
                print('Signal not found. Neither around or after moving. ')
                return -1
            
        d_img = img_processing.inpaint_depth_img(d_img)
        #--------------------------------------------------------------------------#

        #------------------------GEOMETRIC TRANSFORMATIONS-------------------------#
        matrix_3d_points = gt.get_all_3d_points(d_img)
        signal_3d_point = gt.get_single_3d_point(d_img, y_c, x_c)
    
        x_signal = signal_3d_point[0]
        y_signal = signal_3d_point[1]
        signal_distance = np.sqrt((x_signal - 0)**2 + (y_signal - 0)**2) #consider relative distance from robot

        if signal_abs_coords is None:
            if lab_mode == "True":
                print('signal found. updating global signal coordinates')
                signal_abs_coords = gt.update_signal_abs_coords(signal_3d_point, robot_wrapper)

        if signal_distance <= params.STOP_DISTANCE_LIMIT:
            break
        #--------------------------------------------------------------------------#

        #--------------------------MAP CONSTRUCTION--------------------------------#
        #For Locobot Y is positive to left.
        planimetry, robot_coords, signal_coords = map_constructor.construct_planimetry(matrix_3d_points, signal_3d_point)
        if debug == "True":
            plotter.save_planimetry(planimetry, robot_coords, signal_coords, 'raw_planimetry')
            #copy_planimetry = copy.deepcopy(planimetry)
            #plt.imsave('results/raw_planimetry.png', copy_planimetry, cmap='gray', origin='lower')

        planimetry = img_processing.process_planimetry(planimetry, signal_coords)

        if debug == "True":
            plotter.save_image(rgb_img, 'rgb_image', False)
            plotter.save_image(d_img, 'depth_image', True)
            plotter.save_planimetry(planimetry, robot_coords, signal_coords, 'processed_planimetry')
            #copy_planimetry = copy.deepcopy(planimetry)
            #plt.imsave('results/processed_planimetry.png', copy_planimetry, cmap='gray', origin='lower')


        #--------------------------------------------------------------------------#

        #-----------------------------PATH DEFINITION------------------------------#
        start_point = robot_coords

        # chiedere altri***
        end_point_signal = (signal_coords[0], signal_coords[1])
        end_point_path = (signal_coords[0] -15, signal_coords[1])
        
        path = path_planner.compute(planimetry, start_point, end_point_path, False)
        if path is not None:
            #print('Original path: {}'.format(path))
            if debug == "True": 
                plotter.save_planimetry(planimetry, start_point, end_point_signal, 'planimetry_with_trajectory', coords=path)
               

            path = path_planner.shrink_path(path)# To debug yet
            print("SHRINK - ", path)
            if len(path) >= 2:
                path = path_planner.clean_shrink_path(path, end_point_path)
                print("CLEAN - ", path)

            #print('Reduced path: {}'.format(path))
            if debug == "True":
                plotter.save_planimetry(planimetry, start_point, end_point_signal, 'planimetry_with_shrinked_path', coords=path)

            #--------------------------------------------------------------------------#

            #-----------------------------FOLLOW TRAJECTORY---------------------------#
            
            if lab_mode == "True":
                robot_wrapper.follow_trajectory(path, robot_coords)
                time.sleep(3)
                #da sistemare per gestire meglio i thread!!!!!!!!!!!!!!!
                
            
            if signal_abs_coords is not None:
                robot_pose = robot_wrapper.get_robot_position()
                distance = np.sqrt( (robot_pose[0] - signal_abs_coords[0]) ** 2 + (robot_pose[1] - signal_abs_coords[1]) ** 2)
                if distance < 0.30:
                    break
        else:
            print("Path is None!")
            break

    print('Arrived to destination!')

if __name__ == '__main__':
    main()
