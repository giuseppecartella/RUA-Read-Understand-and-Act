import numpy as np
from pyrobot import Robot
from pyrobot.locobot.base_control_utils import (
    get_control_trajectory, 
    get_state_trajectory_from_controls
)

def get_trajectory_rotate_sx(start_pos, dt, r, v, angle):
    w = v / r
    T = int(np.round(angle / w / dt))

    controls = get_control_trajectory("rotate", T, v, w)
    states = get_state_trajectory_from_controls(start_pos, dt, controls)
    return states, controls

def get_trajectory_rotate_dx(start_pos, dt, r, v, angle):
    w = v / r
    T = int(np.round(angle / w / dt))

    controls = get_control_trajectory("negrotate", T, v, w)
    states = get_state_trajectory_from_controls(start_pos, dt, controls)
    return states, controls

def get_trajectory_straight(start_pos, dt, r, v, s):
    w = v / r
    T = abs(int(np.round(s / v / dt)))
    
    controls = get_control_trajectory("straight", T, v, w)
    states = get_state_trajectory_from_controls(start_pos, dt, controls)
    return states, controls

def do_rotation(p, val, dt, r, v):
    if p[1] < 0:
        state, _ = get_trajectory_rotate_dx(val, dt, r, v, abs(np.arctan2(p[1],p[0]+0.000001)))
    else:
        state, _ = get_trajectory_rotate_sx(val, dt, r, v, abs(np.arctan2(p[1],p[0]+0.000001)))
    
    return state

def get_trajectory(bot, path):
    v = bot.configs.BASE.MAX_ABS_FWD_SPEED
    w = bot.configs.BASE.MAX_ABS_TURN_SPEED

    #dt = 1. / (s / v)
    dt = 1. / bot.configs.BASE.BASE_CONTROL_RATE
    r = v / w

    ''''print(bot.configs.BASE.MAX_ABS_FWD_SPEED , bot.configs.BASE.MAX_ABS_TURN_SPEED)
    v = 0.2
    w = 0.5
    exe_time = 4
    dt = 1. / 40
    r = v / w
    bot.base.set_vel(fwd_speed = v, turn_speed = w, exe_time = exe_time)'''

    start_state = np.array(bot.base.get_state("odom"))
    concatenate_state = []

    for i, p in enumerate(path):
        if p[0] == 0.0:
            continue

        print("Iteration: ", i)
        if i == 0:
            val = start_state

            if abs(p[1]) < 0.05:
                concatenate_state, _ = get_trajectory_straight(val, dt, r, v, p[0])
                first_rotation = False
            else:
                state = do_rotation(p, val, dt, r, v)

                diagonal = np.sqrt( p[0] ** 2 + p[1] ** 2 )
                state1, _ = get_trajectory_straight(concatenate_state[-1, :].copy(), dt, r, v, diagonal)
                concatenate_state = np.concatenate([state, state1], 0)
                first_rotation = True
            print("start state: ", start_state)
        else:
            if not first_rotation:
                first_rotation = True
            val = concatenate_state[-1, :].copy()
            
            print("val: ", val)

            if abs(p[1]) < 0.05:
                state, _ = get_trajectory_straight(val, dt, r, v, p[0])
            else:
                state = do_rotation(p, val, dt, r, v)
                print("state after rotation: ", state)

                concatenate_state = np.concatenate([concatenate_state, state], 0)

                diagonal = np.sqrt( p[0] ** 2 + p[1] ** 2 )
                state, _ = get_trajectory_straight(concatenate_state[-1, :].copy(), dt, r, v, diagonal)
                print("state after straight: ", state)
 
            concatenate_state = np.concatenate([concatenate_state, state], 0)

        print("Concatenate state: ", concatenate_state)

    return concatenate_state
    
path1 = [[1.24, -0.01, -0.008064341306767012], [0.26, -0.29, -0.8398896196381794]]
path2 = [[0.74, -0.01, -0.013512691013328216], [0.26, -0.75, -1.2370941502573463], [0.26, -0.01, -0.03844259002118798], [0.24, -0.0, 0.0], [0.26, 0.75, -1.2370941502573463], [0.0, -0.0, 0.0]]
path3 = [[1.08, 0.01, 0.009258994662123083], [0.26, 0.3, 0.8567056281827387], [0.16, 0.03, 0.0]]
path4 = [[0.9, 0.01, 0.011110653897607473], [0.26, 0.37, 0.9582587701845933], [0.26, 0.04, 0.15264932839526515], [0.08, -0.0, 0.0]]
path5 = [[0.45, 0.01, 0.022218565326719057], [0.26, 0.43, 1.0269638704927742], [0.26, 0.06, 0.22679884805388587], [0.18, 0.22, 0.0]]

bot = Robot('locobot')
bot.camera.reset()
states = get_trajectory(bot, path5)
bot.base.track_trajectory(states, close_loop=True)

'''states1, _ = get_trajectory_straight(start_state, dt, r, v, 1.0)
states2, _ = get_trajectory_rotate_dx(states1[-1, :].copy(), dt, r, v, np.pi/2)
states = np.concatenate([states1, states2], 0)
state3, _ = get_trajectory_straight(states[-1, :].copy(), dt, r, v, 0.20)
states = np.concatenate([states, state3], 0)
states4, _ = get_trajectory_rotate_sx(states[-1, :].copy(), dt, r, v, np.pi/2)
states = np.concatenate([states, states4], 0)
state5, _ = get_trajectory_straight(states[-1, :].copy(), dt, r, v, 1.0)
states = np.concatenate([states, state5], 0)   '''
