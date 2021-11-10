# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from obstacle_avoidance import parameters
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

def compute_paths(holes):
    #Lista dei boundaries di gap in cui è possibile passare per andare avanti verso il segnale
    paths = []
    print(holes)
    
    boundary_left = boundary_right = 0
    boundary_left_old = boundary_right_old = -1
    finished = False
    while not finished:
        boundary_left = np.argmin(holes[boundary_left:]) + boundary_left
        boundary_right = np.argmax(holes[boundary_left:]) + boundary_left

        if (boundary_left, boundary_right) == (boundary_left_old, boundary_right_old):
            break

        if boundary_right - boundary_left > 30: #Potrebbe essere un posto per passare -- 10 pixel l'ho stabilito io
            paths.append((boundary_left, boundary_right))

        boundary_left_old = boundary_left
        boundary_right_old = boundary_right
        boundary_left = boundary_right

    return paths

#restituisce le coordinate centrali del gap in cui il robot deve passare.
def compute_different_distance(bot_moves, coordinates_3D_signal, paths, d_img):
    masked_depth = np.where(d_img> 1.0, 0, 255)

    best_distance = -1
    distances = []

    for (x_left_rgb, x_right_rgb) in paths:     
        y_right_rgb = np.where(masked_depth[:, x_right_rgb] == 255)[0][0]
        y_left_rgb = np.where(masked_depth[:, x_left_rgb] == 255)[0][0]
        coordinates_left = compute_3d_point(bot_moves.robot, [x_left_rgb, y_left_rgb], d_img)
        coordinates_right = compute_3d_point(bot_moves.robot, [x_right_rgb, y_right_rgb], d_img)
        print("Boundary_Left - Coordinate 3D dello spazio in cui passare: ", coordinates_left)
        print("Boundary_Rigth - Coordinate 3D dello spazio in cui passare: ", coordinates_right)

        #Calcolo della distanza tra noi e lo spazio in cui passare in modo tale da scegliere la strada più vicina.
        #TODO
        center_x = int(min(coordinates_left[0][0], coordinates_right[0][0]) - parameters.BASE_ROBOT)
        center_y = np.mean([coordinates_left[0][1], coordinates_right[0][1]]).astype(int)

        distances.append([center_x, center_y, 0])
        #Distanza [robot - centro_gap] + [centro_gap - segnale]
        distance = np.sqrt(center_x ** 2 + center_y ** 2) + np.sqrt((center_x - coordinates_3D_signal[0]) ** 2 + (center_y - coordinates_3D_signal[1])**2)

        if best_distance == -1 or distance < best_distance:
            best_distance = distance

    return distances, best_distance
    
    #Return delle coordinate target dove andare.
    '''
    Esempio:
                                          X                X è il segnale
                                                            
                    ******      ********   ****
                             L           R
                                    +                     '+' Noi siamo dove c'è il simbolo                        
    '''

def look_situation_zone(depth_image):
    holes = np.zeros(depth_image_inpaint.shape[1])
    for c in range(depth_image_inpaint.shape[1]):
        results = np.where(depth_image_inpaint[:, c] > 1.0, 0, 255)

        if np.any(results):
            holes[c] = 1 

    return holes

bot = Robot('locobot')
bot.camera.reset()
bot_moves = Robot_Movements_Helper(bot)

rgb_img = cv2.cvtColor(cv2.imread('obstacle3.png'),cv2.COLOR_BGR2GRAY)
template = cv2.imread("utils/template.jpg")
d_img = np.load('obstacle3.npy')
helper_detection = Detection_Helper(template)

found, x_c, y_c = helper_detection.look_for_signal(rgb_img)
x_c = int(x_c)
y_c = int(y_c)

'''image = cv2.circle(rgb_img, (x_c, y_c), 10, (255, 255, 0), 2)
plt.imshow(image)
plt.show()'''

depth_image_inpaint, depth_val_signal = helper_detection.compute_depth_distance(x_c, y_c,d_img)
'''plt.imshow(depth_image_inpaint, cmap='gray')
plt.show()'''

holes = look_situation_zone(depth_image_inpaint)
'''plt.bar(np.arange(depth_image_inpaint.shape[1]), holes)
plt.show()'''

paths = compute_paths(holes)
print('Paths: ', paths)
coordinates_signal = #To compute
target_coordinates = compute_different_distance(bot_moves, coordinates_signal, paths, d_img)

'''if __name__ == '__main__':
    bot = Robot('locobot')
    bot.camera.reset()

    bot_moves = Robot_Movements_Helper(bot)
    template = cv2.imread("utils/template.jpg")
    helper_detection = Detection_Helper(template)
    
    #Keep moving until signal is not found. Each time performs
    ANGLES_RADIANT = np.pi/6
    MAX_ROTATIONS = 13

    found, x_c, y_c = False, None, None
    for i in range(MAX_ROTATIONS):
        print('{} Acquisition of the frame RGBD...'.format(i))
        rgb_img, d_img = bot_moves.read_frame()
        #rgb_img = cv2.cvtColor(cv2.imread('prova.png'), cv2.COLOR_BGR2RGB)
        #d_img = np.load('prova.npy')

        found, x_c, y_c = helper_detection.look_for_signal(rgb_img)
        x_c = int(x_c)
        y_c = int(y_c)

        if found:
            break
        else:
            bot_moves.left_turn(ANGLES_RADIANT)

    if found:
        #signal is found, so now we can manage the robot movement.
        print('Signal found...reaching it!')
        is_arrived = reach_signal_with_obstacle(bot_moves, helper_detection, [x_c, y_c], rgb_img, d_img)
        
        if is_arrived:
            print('Robot arrived to destination...In front of the signal!')
        else:
            print('Something went wrong! Robot no longer sees the signal!')
    else:
        #print('Stop signal NOT FOUND in the neighborhood... Hence, robot will not move!')
        #In questo caso dovremmo muoverci per cercarlo. Io direi di farlo andare in direzione in cui abbiamo massima depth'''
