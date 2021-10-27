import os
import cv2
from pyrobot import Robot
import pyrobot.utils.util as prutil
import numpy as np
from utils.robot_functions import Robot_Movements_Helper
from utils.signal_detection import Detection_Helper

def check_presence_signal_around(bot_moves, helper_detection, stop_image):
    
    for _ in range(6):
        print('Acquisition of the frame RGBD...')
        rgb_img, d_img = bot_moves.read_frame()

        found, _, _ = helper_detection.look_for_signal(rgb_img, stop_image)
        if found == True:
            return True
        bot_moves.left_turn()

    return False

def go_until_stop(bot, bot_moves, helper_detection, stop_image):
    
    while True: 
        #check
        print('Acquisition of the frame RGBD...')
        rgb_img, d_img = bot_moves.read_frame()

        found, x_c, y_c = helper_detection.look_for_signal(rgb_img, stop_image)
        if found == True:
            depth = helper_detection.compute_depth_distance(x_c, y_c)

            # Computing 3D point, passing rows and cols
            # --> da cambiare 1000.0 in 1.0????
            cam_class = bot.camera # si chiamava cosi????
            trans, rot, T = cam_class.get_link_transform(cam_class.cam_cf, cam_class.base_f)
            base2cam_trans = np.array(trans).reshape(-1, 1)
            base2cam_rot = np.array(rot)
            
            # pix_to_3dpt(dept img, rows, cols)
            pts_in_cam = prutil.pix_to_3dpt(d_img, [y_c], [x_c], bot.camera.get_intrinsics(), 1000.0)
            
            pts = pts_in_cam[:3, :].T

            pts = np.dot(pts, base2cam_rot.T)
            pts = pts + base2cam_trans.T
           
            print("3D point as [x, y, z]: " , pts_in_cam)
            # dobbiamo inserire rotazione --> calcolo angolo 
            thetha = ...
            # if sull angolo 
            # bot_moves.left o right
            if depth >= 2.0:
                bot_moves.forward(1.0)  #We continually moving with step = 1 m
                # aggiorniamo la depth
                depth -= 1.0
            else: # sono sufficientemente vicino
                bot_moves.forward(depth)
                return True
        else:
            return False # POI DOVREMMO FARLO GIRARE SE NON LO VEDE PIU????

if __name__ == '__main__':
    
    bot = Robot('locobot')
    bot.camera.reset()

    bot_moves = Robot_Movements_Helper(bot)
    helper_detection = Detection_Helper()
    stop_image = cv2.imread("utils/template.jpg")

    #found = check_presence_signal_around(bot_moves, helper_detection, stop_image)
    for i in range(6):
        print('{} Acquisition of the frame RGBD...'.format(i))
        rgb_img, d_img = bot_moves.read_frame()

        found, _, _ = helper_detection.look_for_signal(rgb_img, stop_image)
        if not found:
            bot_moves.left_turn()


    if found == True:
        
        print('Signal FOUND!')
        arrived = go_until_stop(bot, bot_moves, helper_detection, stop_image)
        
        if arrived == True:
            print('Robot arrived to destination...In front of the signal!')
        else:
            print('Something went wrong! Robot no longer sees the signal!')
    else:
        print('Stop Signal NOT FOUND in the neighborhood... Hence, It will not move!')
