# -*- coding: utf-8 -*-
import numpy as np

from utils.movement_helper import Movement_helper
from .geometry_transformation import GeometryTransformation
import matplotlib.pyplot as plt
from utils.plotter import Plotter
import time
from . import project_parameters as params
import random

class RobotWrapper():
    def __init__(self, lab_mode="True"):
        self.lab_mode = lab_mode
        self.robot = None

        if self.lab_mode == "True":
            from pyrobot import Robot
            self.robot = Robot('locobot')
            self.camera = self.robot.camera

    def explore(self, signal_detector, map_constructor, img_processing, gt, path_planner, signal_abs_coords=None, last=None):  
        plotter = Plotter('results')   
       
        for i in range(params.MAX_ROTATIONS):
            print('{} Rotation ...'.format(i))
            rgb_img, d_img = self.get_rgbd_frame()

         
            found_written_sign, prediction = signal_detector.look_for_written_signal(rgb_img, d_img, self)
            if found_written_sign:
                step_forward = self.get_values_for_action(prediction)
                return None,None,None,None, step_forward

       
            found, x_c, y_c = signal_detector.look_for_signal(rgb_img)

            plt.imsave('results/rgb_image.png', rgb_img)
            plt.imsave('results/depth_image.png', d_img, cmap='gray')

            if found:
                return rgb_img, d_img, x_c , y_c, False
            else:
                self.turn(params.ANGLES_RADIANT)
        

        if last is None:
            # if I am here no signal Found
            EXPLORATION_TIMES = 4
            for times in range(EXPLORATION_TIMES):
                if signal_abs_coords is not None:
                    angle_movement = self.allineate_robot(signal_abs_coords)
                    self.turn(angle_movement)
                
                self.reset_camera()
                rgb_img, d_img = self.get_rgbd_frame()    

                found_written_sign, prediction = signal_detector.look_for_written_signal(rgb_img, d_img, self)
                if found_written_sign:
                    step_forward = self.get_values_for_action(prediction)
                    return None,None,None,None, step_forward       

                found, x_c, y_c = signal_detector.look_for_signal(rgb_img)
                

                plt.imsave('results/rgb_image.jpg', d_img)
                plt.imsave('results/depth_image.png', d_img, cmap='gray')
                if found:
                    return rgb_img, d_img, x_c , y_c, False
                else:
                    # start exploring
                    print('{} Exploration ...'.format(times))
                    d_img = img_processing.inpaint_depth_img(d_img)
                    matrix_3d_points = gt.get_all_3d_points(d_img)

                    MAX_STEP_EXPLORATION = 120
                    # signal False means we do not use the signal coords for the planimetry
                    planimetry, robot_coords , _= map_constructor.construct_planimetry(matrix_3d_points, signal_3d_point=None, signal = False)
                    planimetry = img_processing.process_planimetry(planimetry)
                    
                    start = robot_coords
                    end =  (MAX_STEP_EXPLORATION, robot_coords[1])
                    
                    plotter.save_planimetry(planimetry, robot_coords, end, 'exploring_planimetry')           
                    
                    path = path_planner.compute(planimetry, start , end)
                    plotter.save_planimetry(planimetry, robot_coords, end, 'explore_plan_with_trajectory', coords=path)

                    if (path is not None):
                        path = path_planner.shrink_path(path)
                        if len(path) >= 2:
                            path = path_planner.clean_shrink_path(path, end)
                        self.follow_trajectory(path, robot_coords)
                    else:
                        self.turn(random.uniform(-1.57, 1.57))
        
        return None, None, None, None, None
                    
       
    def get_rgbd_frame(self):
        rgb_img, depth_img = self.camera.get_rgb_depth()
        return rgb_img, depth_img
    
    def reset_camera(self):
        self.robot.camera.reset()

    def get_intrinsic_matrix(self):
        return self.camera.get_intrinsics()

    
    def allineate_robot(self, signal_abs_coords):
        current_pose = self.get_robot_position()

        delta_x = signal_abs_coords[0] - current_pose[0]
        delta_y = signal_abs_coords[1] - current_pose[1]

        alpha = np.arctan2(delta_y, delta_x)
        yaw = current_pose[-1]

        #consider alpha and yaw always positive
        alpha = alpha if alpha >=0 else (2*np.pi + alpha)
        yaw = yaw if yaw >= 0 else (2*np.pi + yaw)

        return alpha - yaw


    def reach_relative_point(self, x, y, theta=0.0):
        target_position = [x, y, theta]
        self.robot.base.go_to_relative(target_position, smooth=False, close_loop=True)

    def reach_absolute_point(self, x, y, theta=0.0):
        target_position = [x, y, theta]
        self.robot.base.go_to_absolute(target_position, smooth=False, close_loop=True)

    def turn(self, angle_radiant):
        target_position = [0.0, 0.0, angle_radiant]
        self.robot.base.go_to_relative(target_position, smooth=False, close_loop=True)
    
    def get_robot_position(self):
        """
        This function returns coordinates of the robot w.r.t. 
        the origin of the global frame
        """
        return self.robot.base.get_state('odom')
    
    def follow_trajectory(self, path, robot_coords):
        movement_helper = Movement_helper()
        states = movement_helper.follow_trajectory(self.robot, path, robot_coords)
        self.robot.base.track_trajectory(states, close_loop=True, wait=True)

    def get_values_for_action(self, prediction):
        if prediction == 2:
            self.turn(1.57) #in radiant
        elif prediction == 3:
            self.turn(-1.57)
        elif prediction == 4:
            time.sleep(5)