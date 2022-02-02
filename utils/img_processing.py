# -*- coding: utf-8 -*-
import numpy as np
import cv2
from utils import project_parameters as params
from utils.map_constructor import MapConstructor

class ImgProcessing():
    def __init__(self):
        pass

    def inpaint_depth_img(self, depth_img):
        result = depth_img.astype(np.single)
        mask = np.where(result == 0, 255, 0).astype(np.ubyte)
        kernel = np.ones((5,5))
        mask = cv2.dilate(mask, kernel, iterations=2)
        result = cv2.inpaint(result, mask, 3, cv2.INPAINT_TELEA)
        result = cv2.medianBlur(result, 5)
        result = cv2.medianBlur(result, 5)
        return result

    def process_planimetry(self, planimetry, signal_coords = None):
        kernel = np.ones((3,3))
        map_constructor = MapConstructor()
        planimetry = cv2.medianBlur(planimetry.astype(np.ubyte), 5)
        radius = 30

        if signal_coords is not None:
            planimetry = map_constructor.circle_around_signal(planimetry, signal_coords[0], signal_coords[1], radius)
        planimetry = cv2.dilate(planimetry, kernel, iterations=1)
        planimetry = cv2.GaussianBlur(planimetry, params.GAUSSIAN_FILTER_SIZE, (params.GAUSSIAN_FILTER_SIZE[0]-1)/5) # filtro 51 sta almeno a 20 cm da ostacoli
        planimetry = np.where(planimetry > 1.9, 255, 0)
        
        return planimetry
