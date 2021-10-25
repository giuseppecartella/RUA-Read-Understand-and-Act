import numpy as np

class Robot_Movements_Helper():
    def __init__(self, robot) -> None:
        self.robot = robot

    def read_frame(self):
        return self.robot.camera.get_rgb_depth()

    def perform_action(self, action_idx, meters=0.25):
        if action_idx == 0:
            target_position = [meters, 0.0, 0.0]
        if action_idx == 1:
            target_position = [0.0, 0.0, np.pi / 3]
        if action_idx == 2:
            target_position = [0.0, 0.0, -np.pi / 3]
        self.robot.base.go_to_relative(target_position, smooth=False, close_loop=True)

    def forward(self, meters):
        self.perform_action(self.robot, 0, meters)

    def left_turn(self):
        self.perform_action(self.robot, 1)

    def right_turn(self):
        self.perform_action(self.robot, 2)
    
    #??? A cosa serve? <-----
    def reset_angle(self):
        for _ in range(3):
            self.left_turn(self.robot)
