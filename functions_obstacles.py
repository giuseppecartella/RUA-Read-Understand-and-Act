# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from obstacle_avoidance import parameters
import pyrobot.utils.util as prutil
from utils.robot_movements import Robot_Movements_Helper
from utils.signal_detection import Detection_Helper

def compute_3d_point(bot,poi, d_img):
    """
        Input:
            -poi : Stands for "point of interest". It is a list [x_c, y_c]
    """
    x_c, y_c = poi[0], poi[1]
    camera = bot.camera
    trans, rot, T = camera.get_link_transform(camera.cam_cf, camera.base_f)
    base2cam_trans = np.array(trans).reshape(-1, 1)
    base2cam_rot = np.array(rot)
    pts_in_cam = prutil.pix_to_3dpt(d_img, [y_c], [x_c], camera.get_intrinsics(), 1.0) 
    pts = pts_in_cam[:3, :].T
    pts = np.dot(pts, base2cam_rot.T)
    pts = pts + base2cam_trans.T
    return pts


"""
def compute_3d_point(bot, poi, d_img):
    x_c, y_c = poi[0], poi[1]

    intrinsic_matrix = np.array([[613.19714, 0.0, 314.70608],
                                 [0.0, 613.91461182, 237.9289],
                                 [0.0,0.0,1.0]])
                            
    point_3d = intrinsic_matrix @ np.array([x_c, y_c, 1])
    return point_3d
"""

def construct_planimetry(final_depth):
    # while raggiunto segnale o numero esplorazioni, fai foto
    NUM_SLICES = 1

    planimetry = np.zeros((NUM_SLICES, final_depth.shape[1]))
    #SLICE_EXTENSION = depth_val_signal/(NUM_SLICES - 1)
    SLICE_EXTENSION = 0.5 #meters

    # senno usiamo solo la prima fascia, andiamo avanti finche possiamo, poi foto di nuovo
    for i in range(NUM_SLICES):
        for c in range(final_depth.shape[1]):
            pos = np.logical_and(final_depth[:,c] > SLICE_EXTENSION*i, final_depth[:,c] <= SLICE_EXTENSION*(i+1))
            
            if np.any(pos):
                planimetry[i][c] = 1
    return planimetry

def concat_paths(paths):
    new_paths = []

    cur_path = paths[0]
    for idx, path in enumerate(paths):
        if idx == len(paths) - 1:
            break

        if paths[idx+1][0] - cur_path[1] < 15:
            cur_path[1] = paths[idx+1][0]
        else:
            new_paths.append(cur_path)
            cur_path = path[idx + 1]
            print(cur_path)

    return new_paths

def compute_paths(holes):
    #Lista dei boundaries di gap in cui è possibile passare per andare avanti verso il segnale
    paths = []
    
    boundary_left = boundary_right = 0
    boundary_left_old = boundary_right_old = -1
    finished = False
    while not finished:
        boundary_left = np.argmin(holes[boundary_left:]) + boundary_left
        boundary_right = np.argmax(holes[boundary_left:]) + boundary_left

        if boundary_left == boundary_right:
            boundary_right = len(holes) - 1

        if (boundary_left, boundary_right) == (boundary_left_old, boundary_right_old):
            break

        print('Boundaries: {}, {}'.format(boundary_left, boundary_right))

        if boundary_right - boundary_left > 10: #30 intanto per provare a scremare i gap troppo piccoli in cui il robot comunque 
                                                #non riuscirà a passare
            paths.append((boundary_left, boundary_right))

        boundary_left_old = boundary_left
        boundary_right_old = boundary_right
        boundary_left = boundary_right
    return paths #lista di tuple, in cui ogni tupla è il margine sx e dx di ogni hole

#restituisce le coordinate centrali del gap in cui il robot deve passare.
def compute_different_distance(bot_moves, coordinates_3D_signal, paths, d_img):
    masked_depth = np.where(d_img> 0.5, 0, 255)
    cv2.imwrite('masked.jpg', masked_depth)

    best_distance = -1
    target_position = [0.0, 0.0, 0.0]
    distances = []

    for (x_left_rgb, x_right_rgb) in paths:    
        print(x_left_rgb, x_right_rgb)
        if x_left_rgb == 0:
            y_right_rgb = np.where(masked_depth[:, x_right_rgb - 1] == 255)[0][0]
            #coordinates_right = compute_3d_point(bot_moves.robot, [x_right_rgb, y_right_rgb], d_img)
            y_left_rgb = y_right_rgb
        elif x_right_rgb == len(d_img[0]) - 1:
            print('Sono entrato')
            y_left_rgb = np.where(masked_depth[:, x_left_rgb - 1] == 255)[0][0]
            print(y_left_rgb)
            #coordinates_left = compute_3d_point(bot_moves.robot, [x_left_rgb, y_left_rgb], d_img)
            y_right_rgb = y_left_rgb
        else:
            y_right_rgb = np.where(masked_depth[:, x_right_rgb - 1] == 255)[0][0]
            y_left_rgb = np.where(masked_depth[:, x_left_rgb - 1] == 255)[0][0]
            

        #print('Eccociiii')
        #print(y_right_rgb, y_left_rgb)
        #print("Boundary_Left - Coordinate 3D dello spazio in cui passare: ", coordinates_left)
        #print("Boundary_Rigth - Coordinate 3D dello spazio in cui passare: ", coordinates_right)
        coordinates_left = compute_3d_point(bot_moves.robot, [x_left_rgb, y_left_rgb], d_img)
        coordinates_right = compute_3d_point(bot_moves.robot, [x_right_rgb, y_right_rgb], d_img)

        gap_space = np.sqrt((coordinates_left[0][0] - coordinates_right[0][0])**2 + (coordinates_left[0][1] - coordinates_right[0][1])**2)
        if gap_space < parameters.BASE_ROBOT + 0.10: #10 cm of tollerance
            continue

        #Calcolo della distanza tra noi e lo spazio in cui passare in modo tale da scegliere la strada più vicina.
        center_x = int(min(coordinates_left[0][0], coordinates_right[0][0]) - parameters.BASE_ROBOT)
        center_y = np.mean([coordinates_left[0][1], coordinates_right[0][1]]).astype(int)

        distances.append([center_x, center_y, 0])
        #Distanza [robot - centro_gap] + [centro_gap - segnale]
        distance = np.sqrt(center_x ** 2 + center_y ** 2) + np.sqrt((center_x - coordinates_3D_signal[0][0]) ** 2 + (center_y - coordinates_3D_signal[0][1])**2)

        if best_distance == -1 or distance < best_distance:
            best_distance = distance
            target_position = [0.5,  - coordinates_left[0][1] - 0.10, 0.0]
    return distances, target_position, best_distance
    
    '''
    Esempio:
                                          X                X è il segnale
                                                            
                    ******      ********   ****
                             L           R
                                    +                     '+' Noi siamo dove c'è il simbolo                        
    '''