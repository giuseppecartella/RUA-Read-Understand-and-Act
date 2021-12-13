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
import copy
import matplotlib.pyplot as plt


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
    plotter = Plotter('results')
    #--------------------------------------------------------------------------#

    signal_abs_coords = None

    while True:
        print('entrato nel whileee')
        #-------------------------------SIGNAL DETECTION---------------------------#
        if lab_mode == "True":
            print('qua')
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
            print('signal found. updating global signal coordinates')
            signal_abs_coords = gt.update_signal_abs_coords(signal_3d_point, robot_wrapper)

        if signal_distance <= params.STOP_DISTANCE_LIMIT:
            break
        #--------------------------------------------------------------------------#

        #--------------------------MAP CONSTRUCTION--------------------------------#
        #For Locobot Y is positive to left.
        planimetry, robot_coords, signal_coords = map_constructor.construct_planimetry(matrix_3d_points, signal_3d_point)
        if debug == "True":
            #plotter.save_planimetry(planimetry, robot_coords, signal_coords, 'raw_planimetry')
            copy_planimetry = copy.deepcopy(planimetry)
            plt.imsave('results/raw_planimetry.png', copy_planimetry, cmap='gray', origin='lower')

        planimetry = img_processing.process_planimetry(planimetry)

        if debug == "True":
            plotter.save_image(rgb_img, 'rgb_image', False)
            plotter.save_image(d_img, 'depth_image', True)
            #plotter.save_planimetry(planimetry, robot_coords, signal_coords, 'processed_planimetry')
            copy_planimetry = copy.deepcopy(planimetry)
            plt.imsave('results/processed_planimetry.png', copy_planimetry, cmap='gray', origin='lower')


        #quantized_planimetry = img_processing.quantize(planimetry, params.QUANTIZATION_WINDOW_SIZE, params.THRESHOLD)
        #Ricordarsi np.where con costante 1.9 da commentare nel caso in cui vogliamo provare con quantizzazione

        #Quantization part
        """start_quantized = img_processing.from_init_to_quantized_space(start, params.QUANTIZATION_WINDOW_SIZE)
        end_quantized = img_processing.from_init_to_quantized_space(end, params.QUANTIZATION_WINDOW_SIZE)
        #plotter.save_planimetry(quantized_planimetry, start_quantized, end_quantized, 'quantized_img')
        
        path = path_planner.compute(quantized_planimetry, start_quantized, end_quantized, False)
        
        for i in path:
            quantized_planimetry[i[0], i[1]] = 255
            coord = img_processing.from_quantize_space_to_init(i, params.QUANTIZATION_WINDOW_SIZE)
            planimetry[coord[0], coord[1]] = 255
            
        
        plotter.save_planimetry(planimetry, start, end, 'path_img_nicholas')
        #plotter.save_planimetry(quantized_planimetry, start_quantized, end_quantized, 'path_img_quantized')
        """
        #--------------------------------------------------------------------------#

        #-----------------------------PATH DEFINITION------------------------------#
        start_point = robot_coords
        end_point = (signal_coords[0], signal_coords[1])
        path = path_planner.compute(planimetry, start_point, end_point, False)
        print('Original path: {}'.format(path))
        if debug == "True":
            #plotter.save_planimetry(planimetry, start_point, end_point, 'planimetry_with_trajectory', coords=path)
            copy_planimetry = copy.deepcopy(planimetry)

            for i in range(len(path)):
                x = path[i][0]
                y = path[i][1]
                copy_planimetry[x,y] = 100
            plt.imsave('results/planimetry_with_trajectory.png', copy_planimetry, cmap='gray', origin='lower')

        path = path_planner.shrink_path(path)# To debug yet
        print('Reduced path: {}'.format(path))
        if debug == "True":
            #plotter.save_planimetry(planimetry, start_point, end_point, 'planimetry_with_shrinked_path', coords=path)
            copy_planimetry = copy.deepcopy(planimetry)

            for i in range(len(path)):
                x = path[i][0]
                y = path[i][1]
                copy_planimetry[x,y] = 100
            plt.imsave('results/planimetry_with_shrinked_path.png', copy_planimetry, cmap='gray', origin='lower')

        #--------------------------------------------------------------------------#

        #-----------------------------FOLLOW TRAJECTORY---------------------------#
        if lab_mode == "True":
            angular_path = robot_wrapper.follow_trajectory(path, robot_coords)
            print("------\nAngular\n")
            print(angular_path)
            break

        """
        codice da testare per capire se funziona l'aggiornamento della global position
        if lab_mode == "True":
            robot_wrapper.follow_trajectory_with_update(path, robot_coords) #global position is updated inside the function
        """

    print('Arrived to destination!')

if __name__ == '__main__':
    main()
