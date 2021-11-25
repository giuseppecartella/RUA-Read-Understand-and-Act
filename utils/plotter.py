# -*- coding: utf-8 -*-
import os
import matplotlib.pyplot as plt
import numpy as np

class Plotter():
    def __init__(self, root_directory):
        self.root_directory = root_directory

    def save_image(self, img, title, is_single_channel=True):
        filename = os.path.join(self.root_directory, title + '.png')
        if is_single_channel:
            plt.imsave(filename, img, cmap='gray')    
        else:
            plt.imsave(filename, img)

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
