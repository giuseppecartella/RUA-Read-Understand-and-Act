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

    def process_planimetry(self, planimetry):
        kernel = np.ones((3,3))
        # serve per togliere quei puntini bianchi ?? --> da controllare in lab 
        planimetry = cv2.medianBlur(planimetry.astype(np.ubyte), 5)
        planimetry = cv2.dilate(planimetry, kernel, iterations=1)
        planimetry = cv2.GaussianBlur(planimetry, (51,51), (51-1)/5) # filtro 51 sta almeno a 20 cm da ostacoli
        return planimetry

    def quantize(self, planimetry, kernel, threshold_min):
        stride = kernel[0]
        
        oW = ((planimetry.shape[1] - kernel[0]) // stride) + 1
        oH = ((planimetry.shape[0] - kernel[1]) // stride) + 1

        out = np.zeros((oH, oW))

        for i in range(oH):
            for j in range(oW):
                out[i,j] = np.count_nonzero(planimetry[stride*i:stride*i+kernel[0], stride*j:stride*j+kernel[1]])
        
        out = np.where(out > threshold_min, 255, 0)
        return out.astype('uint8')

    def from_quantize_space_to_init(self, coords_point, kernel):
        x = coords_point[0] * kernel[0] + (kernel[0] // 2)
        y = coords_point[1] * kernel[1] + (kernel[1] // 2)
        return (x,y)

    def from_init_to_quantized_space(self, coords_point, kernel):
        x = coords_point[0] // kernel[0]
        y = coords_point[1] // kernel[1]
        return (x,y)
