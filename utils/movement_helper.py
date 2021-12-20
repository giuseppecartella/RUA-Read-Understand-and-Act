# -*- coding: utf-8 -*-
import numpy as np

from pyrobot import Robot
from utils.geometry_transformation import GeometryTransformation
from pyrobot.locobot.base_control_utils import (
    get_control_trajectory, 
    get_state_trajectory_from_controls
)

class Movement_helper():
    def get_trajectory_rotate_sx(self, start_pos, dt, r, v, angle):
        w = v / r
        T = int(np.round(angle / w / dt))

        controls = get_control_trajectory("rotate", T, v, w)
        states = get_state_trajectory_from_controls(start_pos, dt, controls)
        return states, controls

    def get_trajectory_rotate_dx(self, start_pos, dt, r, v, angle):
        w = v / r
        T = int(np.round(angle / w / dt))

        controls = get_control_trajectory("negrotate", T, v, w)
        states = get_state_trajectory_from_controls(start_pos, dt, controls)
        return states, controls

    def get_trajectory_straight(self, start_pos, dt, r, v, s):
        w = v / r
        T = abs(int(np.round(s / v / dt)))
        
        controls = get_control_trajectory("straight", T, v, w)
        states = get_state_trajectory_from_controls(start_pos, dt, controls)
        return states, controls

    def do_rotation(self, p, val, dt, r, v):
        if p[1] < 0:
            state, _ = self.get_trajectory_rotate_dx(val, dt, r, v, abs(np.arctan2(p[1],p[0]+0.000001)))
        else:
            state, _ = self.get_trajectory_rotate_sx(val, dt, r, v, abs(np.arctan2(p[1],p[0]+0.000001)))
        
        return state

    def follow_trajectory(self, bot, trajectory, starting_pose):
            starting_3D = np.array(bot.base.get_state("odom"))
            previous_point = starting_pose
            last_concatenate_state = starting_3D.copy()
            previous_yaw = starting_3D[-1].copy()
            concatenate_state = [starting_3D]

            gt = GeometryTransformation()
            for idx, i in enumerate(range(len(trajectory))):
                current_yaw = last_concatenate_state[-1]
                delta_x = trajectory[i][0] - previous_point[0]
                delta_y = trajectory[i][1] - previous_point[1] 
                delta_yaw = current_yaw - previous_yaw #to test if it is the correct way
                rotated_point = gt.rotate_point(delta_x, delta_y, delta_yaw)

                x = rotated_point[0] / 100
                y = rotated_point[1] / 100
                theta = np.arctan2(y,x)

                if idx == (len(trajectory) - 1):
                    theta = 0.0

                destination = [x, y, theta]
                previous_yaw = current_yaw
                concatenate_state = self.get_trajectory(bot, destination, i, concatenate_state)
                last_concatenate_state = concatenate_state[-1].copy()

                print('X: {}, Y:{}, THETA:{}'.format(x,y,theta))
                previous_point = trajectory[i]
            
            return concatenate_state

    def get_trajectory(self, bot, dest, i, concatenate_state):
        v = bot.configs.BASE.MAX_ABS_FWD_SPEED
        w = bot.configs.BASE.MAX_ABS_TURN_SPEED

        dt = 1. / bot.configs.BASE.BASE_CONTROL_RATE
        r = v / w

        if dest[0] == 0.0:
            return concatenate_state

        #print("Iteration: ", i)

        print(concatenate_state[-1])
        val = concatenate_state[-1].copy()
        #print("Val: ", val)

        if i == 0:
            if abs(dest[1]) < 0.05:
                concatenate_state, _ = self.get_trajectory_straight(val, dt, r, v, dest[0])
            else:
                state = self.do_rotation(dest, val, dt, r, v)
                #print("state after rotation: ", state)

                diagonal = np.sqrt( dest[0] ** 2 + dest[1] ** 2 )
                state1, _ = self.get_trajectory_straight(state[-1].copy(), dt, r, v, diagonal)
                #print("state after straight: ", state1)
                concatenate_state = np.concatenate([state, state1], 0)
        else:
            if abs(dest[1]) < 0.05:
                state, _ = self.get_trajectory_straight(val, dt, r, v, dest[0])
            else:
                state = self.do_rotation(dest, val, dt, r, v)
                #print("state after rotation: ", state)

                concatenate_state = np.concatenate([concatenate_state, state], 0)

                diagonal = np.sqrt( dest[0] ** 2 + dest[1] ** 2 )
                state, _ = self.get_trajectory_straight(concatenate_state[-1].copy(), dt, r, v, diagonal)
                #print("state after straight: ", state)

            concatenate_state = np.concatenate([concatenate_state, state], 0)

        return concatenate_state
