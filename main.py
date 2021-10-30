# -*- coding: utf-8 -*-

import os
import cv2
from pyrobot import Robot
import pyrobot.utils.util as prutil
import numpy as np
from utils.robot_functions import Robot_Movements_Helper
from utils.signal_detection import Detection_Helper

def compute_3d_point(bot, poi):
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

def reach_signal2(bot, bot_moves, helper_detection, first_rgb_img):

    found, x_c, y_c = helper_detection.look_for_signal(first_rgb_img)
    
    is_arrived = False
    while is_arrived:
        if found:
            depth = helper_detection.compute_depth_distance(x_c, y_c)

            pts = compute_3d_point(bot, [x_c, y_c])
            print("3D point as [x, y, z]: " , pts)
            # rotate robot towards the target
            theta = np.arctan2(y_c, x_c)            
            bot_moves.turn(theta)

            if depth >= 2.0:
                bot_moves.forward(1.0)
                rgb_img = bot_moves.read_frame()
                found, x_c, y_c = helper_detection.look_for_signal(rgb_img)
                continue
            else: # sono sufficientemente vicino
                # oppure si puo muovere di [depth, y, 0] cosi si muove in diagonale --> nel caso, fare funzione.
                bot_moves.forward(depth)
                is_arrived = True
                
        else:
            is_arrived = False # POI DOVREMMO FARLO GIRARE SE NON LO VEDE PIU????
            break

    return is_arrived

def reach_signal(bot, bot_moves, helper_detection):
 
    while True: 
        #check
        print('Acquisition of the frame RGBD...')
        rgb_img, d_img = bot_moves.read_frame()

        found, x_c, y_c = helper_detection.look_for_signal(rgb_img)
        if found:
            depth = helper_detection.compute_depth_distance(x_c, y_c)

            pts = compute_3d_point(bot, [x_c, y_c])
            print("3D point as [x, y, z]: " , pts)
            # rotate robot towards the target
            thetha = np.arctan2(y_c, x_c)
            # if sull angolo---> in che senso?
            bot_moves.left_turn(thetha)

            if depth >= 2.0:
                bot_moves.forward(1.0)  #We continually moving with step = 1 m
                # aggiorniamo la depth
                depth -= 1.0 #non è sbagliato aggiornarla qua la depth visto che poi ricomincia il ciclo ma il robot è già andato avanti?
            else: # sono sufficientemente vicino
                bot_moves.forward(depth)
                return True
        else:
            return False # POI DOVREMMO FARLO GIRARE SE NON LO VEDE PIU????



if __name__ == '__main__':
    bot = Robot('locobot')
    bot.camera.reset()
    #bot = 'robot'

    bot_moves = Robot_Movements_Helper(bot)
    template = cv2.imread("utils/template.jpg")
    helper_detection = Detection_Helper(template)
    

    #Keep moving until signal is not found. Each time performs
    ANGLES_RADIANT = np.pi/3
    MAX_ROTATIONS = 6

    found, x_c, y_c = False, None, None
    for i in range(MAX_ROTATIONS):
        print('{} Acquisition of the frame RGBD...'.format(i))
        #rgb_img, d_img = bot_moves.read_frame()
        #substitute for the moment
        rgb_img = cv2.cvtColor(cv2.imread('prova.png'), cv2.COLOR_BGR2RGB)
        d_img = np.load('prova.npy')

        found, x_c, y_c = helper_detection.look_for_signal(rgb_img)

        if found:
            break
        else:
            bot_moves.left_turn(ANGLES_RADIANT)
    
    #NB: FINO A QUA IL CODICE E' STATO TESTATO(PER QUANTO POSSIBILE) DA BEPPE E SEMBRA FUNZIONARE!!

    if found:
        #signal is found, so now we can manage the robot movement.
        print('Signal found...reaching it!')
        is_arrived = reach_signal(bot, bot_moves, helper_detection)
        
        if is_arrived:
            print('Robot arrived to destination...In front of the signal!')
        else:
            print('Something went wrong! Robot no longer sees the signal!')
    else:
        print('Stop signal NOT FOUND in the neighborhood... Hence, robot will not move!')
