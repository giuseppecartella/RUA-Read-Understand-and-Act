import argparse
import cv2
import sys
from matplotlib.pyplot import plot
import numpy as np
from utils import project_parameters as params
from utils import map_constructor
from utils.robot_wrapper import RobotWrapper
from utils.signal_detector import SignalDetector
from utils.img_processing import ImgProcessing
from utils.plotter import Plotter
from utils.geometry_transformation import GeometryTransformation
from utils.map_constructor import MapConstructor
from utils.path_planner import PathPlanner


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
    signal_detector = SignalDetector(lab_mode)
    map_constructor = MapConstructor()
    path_planner = PathPlanner()
    img_processing = ImgProcessing()
    gt = GeometryTransformation()
    plotter = Plotter('results')
    #--------------------------------------------------------------------------#
    
    #-------------------------------SIGNAL DETECTION---------------------------#
    if lab_mode is True:
        rgb_img, d_img = robot_wrapper.get_rgbd_frame()
    else:
        rgb_img = cv2.cvtColor(cv2.imread('test_images/obstacle8.png'), cv2.COLOR_BGR2RGB)
        d_img = np.load('test_images/obstacle8.npy')

    found, x_c, y_c = signal_detector.look_for_signal(rgb_img) #y_c is the row idx, x_c is col_idx
    
    #This condition must be removed when exploration phase is implemented
    if not found:
        print('Signal not found. Impossibile to continue.')
        return -1

    d_img = img_processing.inpaint_depth_img(d_img)
    #--------------------------------------------------------------------------#

    #------------------------GEOMETRIC TRANSFORMATIONS-------------------------#
    matrix_3d_points = gt.get_all_3d_points(d_img)
    signal_3d_point = gt.get_single_3d_point(d_img, y_c, x_c)
    #--------------------------------------------------------------------------#

    #--------------------------MAP CONSTRUCTION--------------------------------#
    #For Locobot Y is positive to left.
    
    planimetry, robot_coords, signal_coords = map_constructor.construct_planimetry(matrix_3d_points, signal_3d_point)
    #planimetry = img_processing.clip_planimetry(planimetry)
    #planimetry = img_processing.process_planimetry(planimetry)
    #--------------------------------------------------------------------------#
    if debug == "True":
        plotter.save_image(rgb_img, 'rgb_image', False)
        plotter.save_image(d_img, 'depth_image', True)
        plotter.save_planimetry(planimetry, robot_coords, signal_coords)

    #POSSIBILE PROBLEMA: nel caso di obstacle 6 eliminare le bande nere è inammissibile
    #xk in questo modo andremmo a tagliare fuori il robot.
    #a mio parere è possibile togliere le bande solo quando middle position
    #è compresa tra boundary left e boundary right.


if __name__ == '__main__':
    main()