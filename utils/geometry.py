# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg.linalg import matrix_power
from . import parameters

def pix_to_3dpt(depth_im, rs, cs, intrinsic_mat, depth_map_factor, reduce=None, k=5):
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

def compute_3d_point(depth_img, rows, cols):
    trans = parameters.trans
    rot = parameters.rot
    intrinsic_matrix = parameters.intrinsic_matrix
    
    base2cam_trans = np.array(trans).reshape(-1, 1)
    base2cam_rot = np.array(rot)
    pts_in_cam = pix_to_3dpt(depth_img, rows, cols, intrinsic_matrix, parameters.DEPTH_MAP_FACTOR) 
    pts = pts_in_cam[:3, :].T
    pts = np.dot(pts, base2cam_rot.T)
    pts = pts + base2cam_trans.T
    return pts

def get_all_3d_points(depth_image):
    rows, cols = depth_image.shape[0], depth_image.shape[1]
    indices = np.indices((rows, cols))
    row_indices = indices[0].reshape((1, -1))[0]
    col_indices = indices[1].reshape((1, -1))[0]
    matrix_3d_points = compute_3d_point(depth_image, row_indices, col_indices).reshape((rows,cols,3))
    return matrix_3d_points

def get_single_3d_point(depth_image, row, col):
    return compute_3d_point(depth_image, row, col)