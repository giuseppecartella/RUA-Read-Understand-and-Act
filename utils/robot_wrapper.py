# -*- coding: utf-8 -*-
import numpy as np
from .geometry_transformation import GeometryTransformation
import matplotlib.pyplot as plt
from utils.plotter import Plotter
import copy
from . import project_parameters as params

class RobotWrapper():
    def __init__(self, lab_mode="True"):
        self.lab_mode = lab_mode
        self.robot = None

        if self.lab_mode == "True":
            from pyrobot import Robot
            self.robot = Robot('locobot')
            self.camera = self.robot.camera

    def explore(self, signal_detector, map_constructor, img_processing, gt, path_planner, signal_abs_coords=None):  
        # TUTTI PARAM IN GRASSETTO DA METTERE POI IN PARAMS QUANDO ABBIAMO DECISO 
        plotter = Plotter('results')
        
        """
        for i in range(params.MAX_ROTATIONS):
            print('{} Rotation ...'.format(i))
            rgb_img, d_img = self.get_rgbd_frame()
            found, x_c, y_c = signal_detector.look_for_signal(rgb_img)

            if found:
                print('SEGNALE TROVATOO')
                plt.imsave('results/rgb.jpg', d_img)
                plt.imsave('results/depth.png', d_img, cmap='gray')
                return rgb_img, d_img, x_c , y_c
            else:
                self.turn(params.ANGLES_RADIANT)
                print("segnale NON trovato")
        """

        # if I am here no signal Found
        EXPLORATION_TIMES = 4
        for times in range(EXPLORATION_TIMES):
            if signal_abs_coords is not None:
                print('conosco dove si trova segnale, mi ruoto')
                angle_movement = self.compute_angle_towards_signal(signal_abs_coords)
                self.turn(angle_movement)
            
            # per evitare di fare una foto uguale, possiamo tenere quella di prima ? --> la vede fuori dal for?
            self.reset_camera()
            rgb_img, d_img = self.get_rgbd_frame()              
            found, x_c, y_c = signal_detector.look_for_signal(rgb_img)
            
            if found:
                plt.imsave('results/rgb.jpg', d_img)
                plt.imsave('results/depth.png', d_img, cmap='gray')
                return rgb_img, d_img, x_c , y_c

            else:
                # start exploring
                print('{} Exploration ...'.format(times))
                d_img = img_processing.inpaint_depth_img(d_img)
                matrix_3d_points = gt.get_all_3d_points(d_img)

                MAX_STEP_EXPLORATION = 120
                # signal False means we do not use the signal coords for the planimetry
                planimetry, robot_coords = map_constructor.construct_planimetry(matrix_3d_points, signal = False)
                planimetry = img_processing.process_planimetry(planimetry)
                
                start = robot_coords
                end =  (MAX_STEP_EXPLORATION, robot_coords[1])
                
                #plotter.save_planimetry(planimetry, robot_coords, end, 'exploring_planimetry')     
                copy_planimetry = copy.deepcopy(planimetry)
                plt.imsave('results/exploring_planimetry.png', copy_planimetry, cmap='gray', origin='lower')       
                
                path = path_planner.compute(planimetry, start , end)
                #plotter.save_planimetry(planimetry, robot_coords, end, 'explore_plan_with_trajectory', coords=path)
                copy_planimetry = copy.deepcopy(planimetry)

                if len(path) > 0:
                    for i in range(len(path)):
                        x = path[i][0]
                        y = path[i][1]
                        copy_planimetry[x,y] = 100
                    plt.imsave('results/planimetry_with_trajectory.png', copy_planimetry, cmap='gray', origin='lower')

                    path = path_planner.shrink_path(path)
                    self.follow_trajectory(path, robot_coords)
                else:
                    if signal_abs_coords is not None:
                        angle_movement = self.compute_angle_towards_signal(signal_abs_coords)
                        self.turn(angle_movement)
                    else:
                        self.turn(- np.pi / 3)
                    
       
    def get_rgbd_frame(self):
        rgb_img, depth_img = self.camera.get_rgb_depth()
        return rgb_img, depth_img
    
    def reset_camera(self):
        self.robot.camera.reset()

    def get_intrinsic_matrix(self):
        return self.camera.get_intrinsics()

    
    def compute_angle_towards_signal(self, signal_abs_coords):
        """
        current_pose = self.get_robot_position()
        print('Current pose: {}'.format(current_pose))
        delta_x = current_pose[0] - signal_abs_coords[0]
        delta_y = current_pose[1] - signal_abs_coords[1]
        yaw = current_pose[-1]
        print('Delta x, y, yaw: {}, {}, {}'.format(delta_x, delta_y, yaw))

        delta_angle = yaw + np.pi/2
        angle_robot_signal = np.arctan2(delta_x, delta_y)
        angle_movement = angle_robot_signal - delta_angle
        print('Delta angle, angle_rob_signal, angle_movement: {},{},{}'.format(delta_angle, angle_robot_signal, angle_movement))
        return angle_movement
        """
        """
        gt = GeometryTransformation()
        current_pose = self.get_robot_position()
        yaw = current_pose[-1]
        dx = current_pose[0]
        dy = current_pose[1]
        x = signal_abs_coords[0]
        y = signal_abs_coords[1]

        translation_matrix = np.array([[1,0,dx],
                                       [0,1,dy],
                                       [0,0, 1]])
        inverse_translation = np.linalg.inv(translation_matrix)

        translated_point = np.matmul(inverse_translation, np.array([x,y,1]).T)

        rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                                    [np.sin(yaw),  np.cos(yaw), 0],
                                    [          0,            0, 1]])
                        
        final_point = np.matmul(rotation_matrix, translated_point.T)

        final_angle = np.arctan2(final_point[0], final_point[1])
        return final_angle
        """
        current_pose = self.get_robot_position()
        x_robot = current_pose[0]
        y_robot = current_pose[1]
        yaw = current_pose[-1]
        x_signal = signal_abs_coords[0]
        y_signal = signal_abs_coords[1]

        delta_x = x_signal - x_robot
        delta_y = y_signal - y_robot

        #now consider the typical coordinates system
        #swap x,y
        delta_x, delta_y = delta_y, delta_x
        #negate new delta_x (i.e. old delta_y)
        delta_x = - delta_x
        
        
        angle = np.arctan2(delta_y, delta_x)

        final_angle = yaw + (np.pi/2) - angle #to verify how yaw is returned by get_state('odom')
        return final_angle


    def reach_relative_point(self, x, y, theta=0.0):
        target_position = [x, y, theta]
        print('Target position: {}'.format(target_position))
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


    def follow_trajectory(self, trajectory, starting_pose):
        starting_yaw = self.get_robot_position()[-1]
        previous_point = starting_pose
        angular_path = []

        gt = GeometryTransformation()
        for idx, i in enumerate(range(len(trajectory))):
            current_yaw = self.get_robot_position()[-1]
            delta_x = trajectory[i][0] - previous_point[0]
            delta_y = trajectory[i][1] - previous_point[1] 
            delta_yaw = current_yaw - starting_yaw #to test if it is the correct way
            rotated_point = gt.rotate_point(delta_x, delta_y, delta_yaw)

            x = rotated_point[0] / 100
            y = rotated_point[1] / 100
            theta = np.arctan2(y,x)

            if idx == (len(trajectory) - 1):
                theta = 0.0
            
            angular_path.append([x, y, theta])

            print('X: {}, Y:{}, THETA:{}'.format(x,y,theta))
            previous_point = trajectory[i]
        
        return angular_path

    """
    def follow_trajectory_with_update(self, trajectory, old_robot_coords):
    
        This function assumes that input trajectory contains points which coordinates 
        refer to a global planimetry
  
        #convert list of tuples to numpy array
        trajectory = np.array(trajectory)
        self._reset_robot_global_position() #reset global robot position to (0,0)
        print('New state:')
        print(self.robot.base.get_state('odom'))

        #update y_coordinates to fit with the new reference system
        trajectory[:,1] -= old_robot_coords[1]
        trajectory[:,0] = -trajectory[:,0]
        print('New path: {}'.format(trajectory))

        #now we have the new x and y coordinates. we can follow the trajectory
        #we compute theta angle in order to avoid robot repositioning each time
        for i in range(len(trajectory)):
            x = trajectory[i,0]
            y = trajectory[i,1]
            theta = np.arctan2(y,x)
            self.reach_absolute_point(x, y, 0.0) #passing theta instead of x,y should avoid robot repositioning!
    """
