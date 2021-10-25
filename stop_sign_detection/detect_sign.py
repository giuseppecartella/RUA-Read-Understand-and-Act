#from pyrobot import Robot
from stop_sign_detector import StopDetector
import cv2
import numpy as np


def main():
    #bot = Robot('locobot')
    #bot.camera.reset()

    stop_detector = StopDetector()

    template = cv2.imread('stop_sign_detection/template.jpg', 0)

    #Acquire and process single frame. To do with many acquired frames?(continuos frame detection?)
    print('Starting acquisition of the frame...')
    
    #rgb, depth = bot.camera.get_rgb_depth()
    rgb = cv2.cvtColor(cv2.imread('stop_sign_detection/prova.png'), cv2.COLOR_BGR2RGB)
    depth = np.load('stop_sign_detection/prova.npy')


    WINDOW_SIZE = 16
    x_c, y_c = stop_detector.detect_stop_signal(template, rgb)
    distance = stop_detector.compute_distance(depth, [x_c,y_c], WINDOW_SIZE)
    print(distance)

    #compute relative position x,y and angle to move robot towards stop signal.
    ...

if __name__ == '__main__':
    main()