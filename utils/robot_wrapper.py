# -*- coding: utf-8 -*-

#from pyrobot import Robot
import numpy as np
import cv2

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
        kernel = np.ones((5,5))
        mask = cv2.dilate(mask, kernel, iterations=2)
        result = cv2.inpaint(result, mask, 3, cv2.INPAINT_TELEA)
        result = cv2.medianBlur(result, 5)
        result = cv2.medianBlur(result, 5)
        return result

    def get_intrinsic_matrix(self):
        return self.camera.get_intrinsics()

    def reach_relative_point(self, x, y):
        target_position = [x, y, 0.0]
        self.robot.base.go_to_relative(target_position, smooth=False, close_loop=True)

    def reach_absolute_point(self, x, y):
        target_position = [x, y, 0.0]
        self.robot.base.go_to_absolute(target_position, smooth=False, close_loop=True)

    def turn(self, angle_radiant):
        target_position = [0.0, 0.0, angle_radiant]
        self.robot.base.go_to_relative(target_position, smooth=False, close_loop=True)