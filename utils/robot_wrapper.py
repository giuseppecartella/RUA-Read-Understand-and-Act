# -*- coding: utf-8 -*-

#from pyrobot import Robot
import numpy as np
import cv2

#da spiegare a sara e nicolas le modifiche:
#aggiunto metodo per intrinsic matrix, calcolo punto nel 3d space
#poi 2 semplice funzioni e basta uno per spostarsi in punto x,y l'altro solo
#per rotazione e basta.
#prima di ritornare get_rgb_depth() facciamo subit inpainting 
#cosi non dobbiamo più pensarci
#np.single = float32
#np.ubyte = uint8

class RobotWrapper():
    def __init__(self):
        #self.robot = Robot('locobot')
        #self.camera = self.robot.camera
        pass

    def get_rgbd_frame(self):
        rgb_img, depth_img = self.camera.get_rgb_depth()
        depth_img = self._inpaint_depth_img(depth_img)
        return rgb_img, depth_img

    def _inpaint_depth_img(self, depth_img):
        result = depth_img.astype(np.single)
        mask = np.where(result == 0, 255, 0).astype(np.ubyte)
        result = cv2.inpaint(result, mask, 3, cv2.INPAINT_TELEA)
        result = cv2.medianBlur(result, 5)
        result = result[1:-1, 1:-1]
        result = cv2.copyMakeBorder(result, 1, 1, 1, 1, cv2.BORDER_REFLECT)
        return result

    def get_intrinsic_matrix(self):
        return self.camera.get_intrinsics()

    def reach_point(self, x, y):
        target_position = [x, y, 0.0]
        self.robot.base.go_to_relative(target_position, smooth=False, close_loop=True)

    def turn(self, angle_radiant):
        target_position = [0.0, 0.0, angle_radiant]
        self.robot.base.go_to_relative(target_position, smooth=False, close_loop=True)