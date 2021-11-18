# -*- coding: utf-8 -*-
import numpy as np
import cv2

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

    def clip_planimetry(self,  planimetry):
        #da discutere per ciò che riguarda obstacle 6
        pass

    def process_planimetry(self, planimetry):
        kernel = np.ones((3,3))
        # serve per togliere quei puntini bianchi ?? --> da controllare in lab 
        planimetry_obstacles = cv2.medianBlur(planimetry.astype(np.ubyte), 5)
        planimetry_obstacles = cv2.dilate(planimetry_obstacles, kernel, iterations=1)
        planimetry_obstacles = cv2.GaussianBlur(planimetry_obstacles, (51,51), (51-1)/5) # filtro 51 sta almeno a 20 cm da ostacoli
