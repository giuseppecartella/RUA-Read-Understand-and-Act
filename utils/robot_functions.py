import numpy as np

class Robot_Movements_Helper():
    def __init__(self, robot):
        self.robot = robot

    def read_frame(self):
        return self.robot.camera.get_rgb_depth()

    def perform_action(self, action_idx, meters, angle_radiant):
        if action_idx == 0:
            target_position = [meters, 0.0, 0.0]
        if action_idx == 1:
            target_position = [0.0, 0.0, angle_radiant]
        if action_idx == 2:
            target_position = [0.0, 0.0, -angle_radiant]
        
        # routa di qualsiasi angolo, tanto angolo con arcant2 avrà segno
        if action_idx == 3:
            target_position = [0.0, 0.0, angle_radiant]
        #self.robot.base.go_to_relative(target_position, smooth=True, close_loop=True)
        print('oook sto facendo azioneee!')

    def forward(self, meters=0.25):
        self.perform_action(0, meters, None)

    def left_turn(self, angle_radiant=np.pi/3):
        self.perform_action(1, None, angle_radiant)

    def right_turn(self, angle_radiant=np.pi/3):
        self.perform_action(2, None, angle_radiant)
    
    def turn(self, angle_radiant):
        self.perform_action(3, None, angle_radiant)
    
    # to reset robot at the starting position 
    def reset_angle(self):
        for _ in range(3):
            self.left_turn()
