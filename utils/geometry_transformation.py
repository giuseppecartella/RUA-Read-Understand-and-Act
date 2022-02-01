# -*- coding: utf-8 -*-
import numpy as np
from . import project_parameters

class GeometryTransformation():
    def __init__(self):
        pass

    def get_all_3d_points(self, depth_image):
        """
        Returns a 3-dimensional tensor that has a vector representing
        3d coordinates in cm (not meters!) for each (i,j) position.
        """
        rows, cols = depth_image.shape
        indices = np.indices((rows, cols))
        row_indices = indices[0].reshape((1, -1))[0]
        col_indices = indices[1].reshape((1, -1))[0]
        matrix_3d_points = self.compute_3d_point(depth_image, row_indices, col_indices).reshape((rows,cols,3))
        matrix_3d_points = np.round(matrix_3d_points * 100).astype(np.intc)
        return matrix_3d_points

    def get_single_3d_point(self, depth_image, row, col):
        """
        Returns an array representing 3d coordinates (in cm!) 
        of point in position (row,col) in the image.
        """
        signal_3d_point = self.compute_3d_point(depth_image, row, col)[0]
        signal_3d_point = np.round(signal_3d_point * 100).astype(np.intc)
        return signal_3d_point

    def compute_3d_point(self, depth_img, rows, cols):
        trans = project_parameters.trans
        rot = project_parameters.rot
        intrinsic_matrix = project_parameters.intrinsic_matrix
        
        base2cam_trans = np.array(trans).reshape(-1, 1)
        base2cam_rot = np.array(rot)
        pts_in_cam = self._pix_to_3dpt(depth_img, rows, cols, intrinsic_matrix, project_parameters.DEPTH_MAP_FACTOR) 
        pts = pts_in_cam[:3, :].T
        pts = np.dot(pts, base2cam_rot.T)
        pts = pts + base2cam_trans.T
        return pts

    def _pix_to_3dpt(self, depth_im, rs, cs, intrinsic_mat, depth_map_factor, reduce=None, k=5):
        assert isinstance(rs, int) or isinstance(rs, list) or isinstance(rs, np.ndarray)
        assert isinstance(cs, int) or isinstance(cs, list) or isinstance(cs, np.ndarray)
        if isinstance(rs, int):
            rs = [rs]
        if isinstance(cs, int):
            cs = [cs]
        if isinstance(rs, np.ndarray):
            rs = rs.flatten()
        if isinstance(cs, np.ndarray):
            cs = cs.flatten()
        R, C = depth_im.shape
        if reduce == "none" or reduce is None:
            depth_im = depth_im[rs, cs]
        elif reduce == "mean":
            depth_im = np.array(
                [
                    np.mean(
                        depth_im[
                            max(i - k, 0) : min(i + k, R), max(j - k, 0) : min(j + k, C)
                        ]
                    )
                    for i, j in zip(rs, cs)
                ]
            )
        elif reduce == "max":
            depth_im = np.array(
                [
                    np.max(
                        depth_im[
                            max(i - k, 0) : min(i + k, R), max(j - k, 0) : min(j + k, C)
                        ]
                    )
                    for i, j in zip(rs, cs)
                ]
            )
        elif reduce == "min":
            depth_im = np.array(
                [
                    np.min(
                        depth_im[
                            max(i - k, 0) : min(i + k, R), max(j - k, 0) : min(j + k, C)
                        ]
                    )
                    for i, j in zip(rs, cs)
                ]
            )
        else:
            raise ValueError(
                "Invalid reduce name provided, only the following"
                " are currently available: [{}, {}, {}, {}]".format(
                    "none", "mean", "max", "min"
                )
            )

        depth = depth_im.reshape(-1) / depth_map_factor
        img_pixs = np.stack((rs, cs)).reshape(2, -1)
        img_pixs[[0, 1], :] = img_pixs[[1, 0], :]
        uv_one = np.concatenate((img_pixs, np.ones((1, img_pixs.shape[1]))))

        intrinsic_mat_inv = np.linalg.inv(intrinsic_mat)
        uv_one_in_cam = np.dot(intrinsic_mat_inv, uv_one)
        pts_in_cam = np.multiply(uv_one_in_cam, depth)
        pts_in_cam = np.concatenate((pts_in_cam, np.ones((1, pts_in_cam.shape[1]))), axis=0)
        return pts_in_cam


    #forse questa funzione è da eliminare
    """
    def coordinate_projection(self, starting_pose, current_pose):
            delta_x = starting_pose[0]
            delta_y = starting_pose[1]
            yaw = starting_pose[-1]
            rot_matrix = np.array(
                [
                    [np.cos(yaw), -np.sin(yaw), delta_x],
                    [np.sin(yaw), np.cos(yaw), delta_y],
                    [0, 0, 1]
                ])
            
            agent_state = np.expand_dims([current_pose[0], current_pose[1], 1], axis=0).T
            agent_state = np.matmul(np.linalg.inv(rot_matrix), agent_state)
            agent_state[-1] = current_pose[-1] - yaw

            return agent_state.reshape((1, 3))
    """

    def rotate_point(self, delta_x, delta_y, yaw):
        rot_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                               [np.sin(yaw),  np.cos(yaw), 0],
                               [          0,            0, 1]])

        original_point = np.array([delta_x, delta_y, 1])
        rotated_point = np.matmul(rot_matrix, original_point.T)
        rotated_point[1] = -rotated_point[1] #swap y coordinate

        return rotated_point

    def rotation_for_global_coords(self, yaw, original_point):
        rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                            [np.sin(yaw),  np.cos(yaw), 0],
                            [          0,            0, 1]])
                
        return np.matmul(rotation_matrix, original_point.T)

    def translate_point(self, dx, dy, original_point):
        translation_matrix = np.array([[1, 0, dx],
                                       [0, 1, dy],
                                       [0, 0,  1]])
        
        translated_point = np.matmul(translation_matrix, original_point.T)
        return translated_point

    def update_signal_abs_coords(self, signal_3d_point, robot_wrapper):
        delta_x = signal_3d_point[0] / 100.0
        delta_y = signal_3d_point[1] / 100.0
    
        robot_state = robot_wrapper.get_robot_position()
        x = robot_state[0]
        y = robot_state[1]
    
        current_yaw = robot_state[-1]
        yaw = current_yaw

        rotated_point = self.rotation_for_global_coords(yaw, np.array([delta_x, delta_y, 1]))
        global_coords = robot_state + rotated_point
    
        '''print(delta_x, delta_y, x, y, yaw, rotated_point)
        print('Global coords: {}'.format(global_coords))'''
        return global_coords

    