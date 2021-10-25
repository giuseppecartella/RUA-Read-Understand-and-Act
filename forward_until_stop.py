from utils.robot_functions import Robot_Movements_Helper
from utils.signal_detection import Detection_Helper
import os
from pyrobot import Robot
import numpy as np

def check_presence_signal_around(bot_moves, helper_detection, stop_image):
    for _ in range(3):
        print('Acquisition of the frame RGBD...')
        rgb_img, d_img = bot_moves.read_frame()

        found, _, _ = helper_detection.look_for_signal(rgb_img, stop_image)
        if found == True:
            return True
        bot_moves.left()

    return False

def go_until_stop(bot_moves, helper_detection, stop_image):
    while True:
        #check
        print('Acquisition of the frame RGBD...')
        rgb_img, d_img = bot_moves.read_frame()

        found, x_c, y_c = helper_detection.look_for_signal(rgb_img, stop_image)
        if found == True:
            depth = helper_detection.compute_depth_distance(x_c, y_c)

            #To compute 3D point from 2D??  <--- Non sono sicuro
            #point_3D, _ = bot.camera.pix_to_3dpt(x_c, y_c)

            if depth >= 2.5:
                bot_moves.forward(0.1)  #We continually moving with step = 10cm
            else:
                return True
        else:
            return False

if __name__ == '__main__':
    bot = Robot('locobot')
    bot.camera.reset()

    bot_moves = Robot_Movements_Helper(bot)
    helper_detection = Detection_Helper()
    stop_image = np.load(os.path.join('utils', 'utils/template.jpg')).astype('float32')

    found = check_presence_signal_around(bot_moves, helper_detection, stop_image)

    if found == True:
        print('Robot is mooving towards the signal!')
        arrived = go_until_stop(bot_moves, helper_detection, stop_image)
        if arrived == True:
            print('Robot arrived to destination...In front of the signal!')
        else:
            print('Something went wrong! Robot no longer sees the signal!')
    else:
        print('Stop Signal not found in the neighborhood... Hence, It will not move!')
