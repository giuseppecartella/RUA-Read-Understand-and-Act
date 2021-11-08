# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import random as rng
from utils.robot_movements import Robot_Movements_Helper
from utils.signal_detection import Detection_Helper

# togliere tutto ciò che è sopra nostra altezza


rgb_img = cv2.cvtColor(cv2.imread('obstacle1.png'),cv2.COLOR_BGR2GRAY)
template = cv2.imread("utils/template.jpg")
d_img = np.load('obstacle1.npy')
helper_detection = Detection_Helper(template)


found, x_c, y_c = helper_detection.look_for_signal(rgb_img)
x_c = int(x_c)
y_c = int(y_c)

if found:
    print("Signal coords:")
    print(x_c, y_c)

    rgb = cv2.circle(rgb_img, (x_c, y_c), color=(255, 0, 0), radius=7, thickness=-1)
    plt.imshow(rgb)
    plt.show()

    
    # ******** da mettere in una funzione ??
    depth = d_img.astype('float32')
    # INPAINTING
    # define masks. Points where depth is = 0.
    mask = np.where(depth == 0, 255, 0).astype('uint8')  
    dst = cv2.inpaint(depth, mask, 5, cv2.INPAINT_TELEA)
    
    # application median filter and removing borders
    median = cv2.medianBlur(dst, 3)
    removed_border = median[1:-1, 1:-1]
    final_depth = cv2.copyMakeBorder(removed_border, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    
    # *****
    
    depth_val_signal = helper_detection.compute_depth_distance(x_c, y_c,d_img )
    print(depth_val_signal)
    print(final_depth.shape)	# Mask to segment regions with depth less than threshold
    plt.imshow(final_depth, cmap='gray')
    plt.show()

    # while raggiunto segnale o numero esplorazioni, fai foto
    obstacle = np.zeros((1, final_depth.shape[1]))
    third = depth_val_signal/3
    
    # per usare tutte e tre le fasce --> metodo che elimina pavimento
    # senno usiamo solo la prima fascia, andiamo avanti finche possiamo, poi foto di nuovo
    #for i in range(1):
    print("range: ", 0.0 , 1.0)
    for c in range(final_depth.shape[1]):
        pos = np.where(final_depth[:,c] < 1.0)
        
        if np.any(pos):
            obstacle[0][c] = 1        

    plt.bar(np.arange(final_depth.shape[1]), obstacle[0])
    plt.show()
  

    # calcolo 3D coords      ****       ************
    
    masked_depth = np.where(final_depth> 1.0, 0, 255)
    plt.imshow(masked_depth)
    plt.show()

    center_x = final_depth.shape[1]//2

    left = obstacle[0][:center_x]
    right = obstacle[0][center_x:]

    x_right = np.argmax(right) + center_x

    left = left[::-1]
    x_left = center_x - np.argmax(left)

    y_right = np.where(masked_depth[:, x_right] == 255)[0][0]
    y_left = np.where(masked_depth[:, x_left -1] == 255)[0][0]

    # passarle a compute_3d_point --> trovo y --> se ci passa il robot, forward di 1 metro.
 
    

