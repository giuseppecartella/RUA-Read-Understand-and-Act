# -*- coding: utf-8 -*-
import numpy as np
from .geometry_transformation import GeometryTransformation
import matplotlib.pyplot as plt
from utils.plotter import Plotter
import time
from . import project_parameters as params

class RobotWrapper():
    def __init__(self, lab_mode="True"):
        self.lab_mode = lab_mode
        self.robot = None

        if self.lab_mode == "True":
            from pyrobot import Robot
            self.robot = Robot('locobot')
            self.camera = self.robot.camera

    def explore(self, signal_detector, map_constructor, img_processing, gt, path_planner):  
        # TUTTI PARAM IN GRASSETTO DA METTERE POI IN PARAMS QUANDO ABBIAMO DECISO 
        plotter = Plotter()
        for i in range(params.MAX_ROTATIONS):
            print('{} Rotation ...'.format(i))
            rgb_img, d_img = self.get_rgbd_frame()
            found, x_c, y_c = signal_detector.look_for_signal(rgb_img)
            if found:
                plt.imsave('results/rgb.jpg', d_img)
                plt.imsave('results/depth.png', d_img, cmap='gray')
                return rgb_img, d_img, x_c , y_c

        # if I am here no signal Found
        EXPLORATION_TIMES = 4
        for times in range(EXPLORATION_TIMES):
            
            # per evitare di fare una foto uguale, possiamo tenere quella di prima ? --> la vede fuori dal for?
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
                
                plotter.save_planimetry(planimetry, robot_coords, end, 'exploring_planimetry')            
                
                path = path_planner.compute(planimetry, start , end)
                plotter.save_planimetry(planimetry, robot_coords, end, 'explore_plan_with_trajectory', coords=path)

                #path = path_planner.shrink_path(path)
                #self.follow_trajectory_with_update(path, robot_coords)

       
    def get_rgbd_frame(self):
        rgb_img, depth_img = self.camera.get_rgb_depth()
        return rgb_img, depth_img

    def get_intrinsic_matrix(self):
        return self.camera.get_intrinsics()

    def reach_relative_point(self, x, y, theta=0.0):
        target_position = [x, y, theta]
        self.robot.base.go_to_relative(target_position, smooth=False, close_loop=True)

    def reach_absolute_point(self, x, y, theta=0.0):
        target_position = [x, y, theta]
        self.robot.base.go_to_absolute(target_position, smooth=False, close_loop=True)

    def turn(self, angle_radiant):
        target_position = [0.0, 0.0, angle_radiant]
        self.robot.base.go_to_relative(target_position, smooth=False, close_loop=True)

    def _reset_robot_global_position(self):
        self.robot.base.base_state.state.update(0.0,0.0,0.0)

    def follow_trajectory_new(self, trajectory, robot_coords):
        current_point_planimetry = robot_coords

        for i in range(len(trajectory) - 1):
            delta_x = trajectory[i + 1][0] - current_point_planimetry[0]
            delta_y = - (trajectory[i + 1][1] - current_point_planimetry[1])

            #Per calcolare la current pose rispetto al global frame dobbiamo 
            # considerare l'orientamento del robot

    def follow_trajectory(self, trajectory, robot_coords):    
        pose_x, pose_y, pose_yaw = self.robot.base.get_state('odom')
        starting_pose = np.array([pose_x, pose_y, pose_yaw])
        y_robot = robot_coords[1]

     

        old_path = (0,0)

        for i in range(len(trajectory)):
            if old_path[0] == 0 and old_path[1] == 0:
                y_new = trajectory[i][1] / 100.0
            else:
                y_new = (y_robot - trajectory[i][1])/100.0

            #print('Considered coords: {}'.format(trajectory[i]))
            #print('Old path: {}'.format(old_path))
            #print('Starting pose: {}'.format(starting_pose))

            x_new = trajectory[i][0]/100.0
            x_new -= (old_path[0] / 100.0)
            y_new -= ((y_robot - old_path[1]) / 100.0)

            #print(x_new, y_new)
            current_pose = starting_pose + np.array([x_new,y_new,0.0])
            #print('Current pose: {}'.format(current_pose))

            gt = GeometryTransformation()
            coords = gt.coordinate_projection(starting_pose, current_pose)
           
            #print('Final coordinates: {}'.format(coords))
        
            starting_pose = current_pose     
            old_path = trajectory[i]
            
            if abs(coords[0][1]) < 0.1:
                coords[0][1] = 0.0
        
            self.reach_relative_point(coords[0][0], coords[0][1])
        
           



    def follow_trajectory_with_update(self, trajectory, old_robot_coords):
        """
        This function assumes that input trajectory contains points which coordinates 
        refer to a global planimetry
        """
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