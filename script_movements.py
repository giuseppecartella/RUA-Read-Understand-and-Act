from pyrobot import Robot

robot = Robot('locobot')

angle_rotation = 1,5708 #90 gradi in radianti (se positivo robot ruota verso sinistra)
robot.go_to_relative([1.0, 0.0, angle_rotation])