# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from . import project_parameters as params

class MapConstructor():

    def __init__(self):
        pass

    def construct_planimetry(self, matrix_3d_points, signal_3d_point):
        y_left = np.max(matrix_3d_points[:,:,1]) #max because y is positive to left
        y_right = np.min(matrix_3d_points[:,:,1])
        y_range = np.abs(y_left - y_right)
        middle_position = int(np.round(y_range / 2))

        signal_depth = signal_3d_point[0]
        coordinates_X = matrix_3d_points[:,:,0]
        coordinates_Z = matrix_3d_points[:,:,2]

        mask = np.logical_and(coordinates_Z > (params.FLOOR_HEIGHT_LIMIT * 100), coordinates_Z < (params.ROBOT_HEIGHT * 100))
        mask = np.logical_and(mask, coordinates_X < signal_depth)

        obstacles = matrix_3d_points[mask==True]
        x_coords = obstacles[:,0]
        y_coords = middle_position - obstacles[:,1]

        planimetry = np.zeros((signal_depth, y_range))
        planimetry[x_coords, y_coords] = 255
        planimetry = np.where(planimetry < 0, 0, planimetry) #We put 0 for values which can become negative

        robot_coords = [0, middle_position]
        signal_coords = [signal_depth, middle_position - signal_3d_point[1]]

        planimetry = self._clip_planimetry(planimetry, middle_position)
        return planimetry, robot_coords, signal_coords

    def _clip_planimetry(self, planimetry, middle_position):
        boundary_left = np.max(planimetry, axis=0)
        boundary_left = np.argmax(boundary_left)
        flipped_matrix = planimetry[:,::-1]
        boundary_right = np.max(flipped_matrix, axis=0)
        boundary_right = np.argmax(boundary_right)
        boundary_right = planimetry.shape[1] - boundary_right

        if middle_position:
            pass
        
        planimetry = planimetry[:, boundary_left:boundary_right]
