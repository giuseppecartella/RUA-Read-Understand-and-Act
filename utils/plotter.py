# -*- coding: utf-8 -*-
import os
from PIL.Image import new
import matplotlib.pyplot as plt
import numpy as np
import cv2

class Plotter():
    def __init__(self, root_directory):
        self.root_directory = root_directory

    def save_image(self, img, title, is_single_channel=True):
        filename = os.path.join(self.root_directory, title + '.png')
        if is_single_channel:
            plt.imsave(filename, img, cmap='gray')    
        else:
            plt.imsave(filename, img)

    """
    def save_planimetry(self, planimetry, robot_coords, signal_coords, title, coords=None):
        filename = os.path.join(self.root_directory, title +'.png')
        plt.matshow(planimetry, cmap='gray', origin='lower')
        plt.scatter(robot_coords[1], robot_coords[0], c='r')
        plt.scatter(signal_coords[1], signal_coords[0] - 1, c='y') #-1 in order to obtain a better plot
        if coords is not None:
            coords = np.array(coords)
            x_coords = coords[:,0]
            y_coords = coords[:,1]
            plt.scatter(y_coords, x_coords, c='b', marker='.')
        plt.savefig(filename)
    """

    def save_planimetry(self, planimetry, robot_coords, signal_coords, title, coords=None):
        filename = os.path.join(self.root_directory, title +'.png')
        planimetry = cv2.flip(planimetry,0)
        new_planimetry = np.zeros((planimetry.shape[0], planimetry.shape[1], 3))
        new_planimetry[:,:,0] = planimetry
        new_planimetry[:,:,1] = planimetry
        new_planimetry[:,:,2] = planimetry
        height = new_planimetry.shape[0]

        # Red color in BGR
        red = (0, 0, 255)
        green = (0, 255, 0)
        blue = (255, 0, 0)

        x_r, y_r = height - robot_coords[0] - 1, robot_coords[1]
        x_s, y_s = height - signal_coords[0] - 1, signal_coords[1]
        cv2.circle(new_planimetry, (y_r,x_r), 3, color=red, thickness=-1)
        cv2.circle(new_planimetry, (y_s,x_s), 3, color=green, thickness=-1)
        
        
        if coords is not None:
            for point in coords:
                cv2.circle(new_planimetry, (point[1], height - point[0] - 1), 2, color=blue, thickness=-1)

        cv2.imwrite(filename, new_planimetry)
  