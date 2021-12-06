import numpy as np
from pyrobot import Robot
from pyrobot.locobot.base_control_utils import (
    get_trajectory_circle,
    get_trajectory_negcircle,
    get_control_trajectory, 
    get_state_trajectory_from_controls
)

def get_trajectory_rotate_dx(start_pos, dt, r, v, angle):
    #dt = s /v
    #dt = 1.0 / dt
    w = v / r
    T = int(np.round(angle / w / dt))
    controls = get_control_trajectory("rotate", T, v, w)
    states = get_state_trajectory_from_controls(start_pos, dt, controls)
    return states, controls

def get_trajectory_rotate_sx(start_pos, dt,  r, v, angle):
    #dt = s /v
    #dt = 1.0 / dt
    w = v / r
    T = int(np.round(angle / w / dt))
    controls = get_control_trajectory("negrotate", T, v, w)
    states = get_state_trajectory_from_controls(start_pos, dt, controls)
    return states, controls

def get_trajectory_straight(start_pos, dt, r, v, angle):
    #dt = s /v 
    #dt = 1.0 / dt
    w = v / r
    T = int(np.round(angle / w / dt))
    
    controls = get_control_trajectory("straight", T, v, w)
    states = get_state_trajectory_from_controls(start_pos, dt, controls)
    return states, controls


def get_trajectory(bot):
    v = bot.configs.BASE.MAX_ABS_FWD_SPEED
    w = bot.configs.BASE.MAX_ABS_TURN_SPEED

    s = 1.0 #1 meter ahead
    # v = s / t
    # v t = s
    # t = s / v
    dt = 1.0 / (s / v)
    r = v / w
    start_state = np.array(bot.base.get_state("odom"))
    states1, _ = get_trajectory_straight(start_state,dt , r, v, np.pi)
    print(start_state.shape, states1.shape)
    final_states = np.concatenate([start_state, states1], 0)

    return final_states


    """
    s = 2.0
    dt = 1.0 / bot.configs.BASE.BASE_CONTROL_RATE
    
    print("dt :", dt)
    #v = 2.0 / dt
    v = bot.configs.BASE.MAX_ABS_FWD_SPEED
    dt = (v / s)
    w = bot.configs.BASE.MAX_ABS_TURN_SPEED
    r = v / w
    print("R: " ,r)
    print("v: " , v)
    print("w: ", w)
    print("inizio traiettoria")
    start_state = np.array(bot.base.get_state("odom"))
    states1, _ = get_trajectory_straight(start_state,dt , r, v, np.pi)

    states_new, _ = get_trajectory_straight(states1[-1,:].copy(),dt , r, v, np.pi)
    new_concat = np.concatenate([states1, states_new], 0)
    states2, _ = get_trajectory_rotate_dx(new_concat[-1, :].copy(), dt, r, v, np.pi/2)
    
    states2_new = np.concatenate([new_concat, states2], 0)
    s = 1.0
    dt = (v / s)
    #v = 1.0 / dt
    r = v / w
    print("R: " ,r)
    print("v: " , v)
    print("w: ", w)
       

    states3, _ = get_trajectory_straight(states2[-1, :].copy(), dt , r, v , np.pi)
    states2_new = np.concatenate([states2_new, states3], 0)

    #states = np.concatenate([states1, states2], 0)
    states3 = np.concatenate([states2, states3], 0)

    states4, _ = get_trajectory_rotate_sx(states3[-1, :].copy(), dt, r, v, np.pi/2)

    states4 = np.concatenate([states3, states4], 0)
    #v = 5.0 / dt
    r = v/w
    print("R: " ,r)
    print("v: " , v)
    print("w: ", w)

    states5, _ = get_trajectory_straight(states4[-1, :].copy(), dt , r, v , np.pi)

    

    states = np.concatenate([states4, states5], 0)
    print(states)
    #return states
    return states2_new
    """

print("sono entrato")
bot = Robot('locobot')
states = get_trajectory(bot)
bot.base.track_trajectory(states, close_loop=True)