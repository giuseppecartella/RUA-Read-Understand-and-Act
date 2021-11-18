import argparse
import cv2
import numpy as np
from utils import project_parameters as params
from utils.robot_wrapper import RobotWrapper
from utils.signal_detector import SignalDetector
from utils.img_processing import ImgProcessing
from utils.plotter import Plotter
from utils.geometry_transformation import GeometryTransformation
from utils.path_planner import PathPlanner
from utils.map_constructor import MapConstructor


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
    # invertire gli if --> se in lab_mode fai while per routarsi o cercare il segnale
    if lab_mode is True:
        rgb_img, d_img = robot_wrapper.get_rgbd_frame()
    else:
        rgb_img = cv2.cvtColor(cv2.imread('test_images/obstacle5.png'), cv2.COLOR_BGR2RGB)
        d_img = np.load('test_images/obstacle5.npy')

    found, x_c, y_c = signal_detector.look_for_signal(rgb_img) #y_c is the row idx, x_c is col_idx
    
    #This condition must be removed when exploration phase is implemented
    if not found:
        #robot_wrapper.turn(params.ANGLES_RADIANT)
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
    #--------------------------------------------------------------------------#
    if debug == "True":
        pass
        #plotter.save_image(rgb_img, 'rgb_image', False)
        #plotter.save_image(d_img, 'depth_image', True)
        #plotter.save_planimetry(planimetry, robot_coords, signal_coords)
    
    planimetry, robot_coords, signal_coords = map_constructor.construct_planimetry(matrix_3d_points, signal_3d_point)
    planimetry = img_processing.process_planimetry(planimetry)
    plotter.save_planimetry(planimetry, robot_coords, signal_coords , 'planimetry')
    quantized_planimetry = img_processing.quantize(planimetry, params.QUANTIZATION_WINDOW_SIZE, params.THRESHOLD)
    
    
    #Quantizazion phase
 
    start = robot_coords
    end = (signal_coords[0]- 30, signal_coords[1])
    #Same thing with quantized coordinates
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
   
    planimetry_obstacles_aStar = np.where(planimetry > 1.9,255, 0)
    #plotter.save_planimetry(planimetry_obstacles_aStar, start, end, 'path_img_sara_dilated')   
    
    path = path_planner.compute(planimetry_obstacles_aStar, start, end, False)
    print(path)

    for i in path:
        planimetry[i[0], i[1]] = 255
    plotter.save_planimetry(planimetry, start, end, 'path_img_sara')
    
    
    if debug == "True":
        pass
        #plotter.save_image(rgb_img, 'rgb_image', False)
        #plotter.save_image(d_img, 'depth_image', True)
        #plotter.save_planimetry(planimetry, robot_coords, signal_coords)


    #PATH COMPUTATION---------------------------#

    pose_x, pose_y, pose_yaw = robot_wrapper.robot.base.get_state('odom')
    starting_pose = np.array([pose_x, pose_y, pose_yaw])
    middle_position = robot_coords[1]

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
        x_new -= (old_path[0] / 100.0)
        y_new -= ((middle_position - old_path[1]) / 100.0)

        print(x_new, y_new)
        current_pose = starting_pose + np.array([x_new,y_new,0.0])
        print('Current pose: {}'.format(current_pose))
        coords = gt.coordinate_projection(starting_pose, current_pose)
        print('Final coordinates: {}'.format(coords))
     
        starting_pose = current_pose
       
            
        old_path = path[i]
        
        if abs(coords[0][1]) < 0.1:
            coords[0][1] = 0.0
    

        robot_wrapper.reach_relative_point(coords[0][0], coords[0][1])

        # PENSARE A MOVIMENTI PIU FLUIDI, SOLO QUANDO CAMBIA LA Y O LA X

        #ATTENZIONE IN OTTICA DI IMPLEMENTAZIONE DI UN WHILE 
        #CHE QUINDI PERMETTA DI NON DOVER RIAVVIARE OGNI VOLTA LO SCRIPT
        #BISOGNA RIAGGIORNARE LA GLOBAL POSITION SETTANDOLA A ZERO
        #CIO VA FATTO OGNI VOLTA CHE SI RIACQUISISCE
    
            

if __name__ == '__main__':
    main()