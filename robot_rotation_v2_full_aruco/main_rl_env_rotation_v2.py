from vision_utilities import Vision
from motor_utilities import Motor
import random
import cv2
import numpy as np


class RL_ENV:

    def __init__(self):
        self.goal_angle = 0.0
        self.motors_config = Motor()
        self.vision_config = Vision()

    def reset_env(self):
        # move the robot to home position:
        id_1_dxl_home_position = 390
        id_2_dxl_home_position = 350
        id_3_dxl_home_position = 685
        id_4_dxl_home_position = 655

        print("Sending Robot to Home Position")
        self.motors_config.move_motor_step(id_1_dxl_home_position, id_2_dxl_home_position,
                                           id_3_dxl_home_position, id_4_dxl_home_position)

        self.goal_angle = random.randint(-180, 180)
        print("New Goal angle generated", self.goal_angle)


    def state_space_function(self):
        while True:
            state_space_vector, raw_img, detection_status = self.vision_config.calculate_marker_pose()
            if detection_status:
                break
            else:
                print("waiting for camera and marks")
        return state_space_vector, raw_img

    def env_step(self, actions):
        # put the outputs in operation step area
        id_1_dxl_goal_position = (actions[0] - (-1)) * (450 - 420) / (1 - (-1)) + 420
        id_2_dxl_goal_position = (actions[1] - (-1)) * (305 - 205) / (1 - (-1)) + 205
        id_3_dxl_goal_position = (actions[2] - (-1)) * (640 - 610) / (1 - (-1)) + 610
        id_4_dxl_goal_position = (actions[3] - (-1)) * (820 - 720) / (1 - (-1)) + 720

        id_1_dxl_goal_position = int(id_1_dxl_goal_position)
        id_2_dxl_goal_position = int(id_2_dxl_goal_position)
        id_3_dxl_goal_position = int(id_3_dxl_goal_position)
        id_4_dxl_goal_position = int(id_4_dxl_goal_position)

        self.motors_config.move_motor_step(id_1_dxl_goal_position,
                                           id_2_dxl_goal_position,
                                           id_3_dxl_goal_position,
                                           id_4_dxl_goal_position)

    def env_render(self, image, done=False, step=1, episode=1, cylinder=0):

        if done:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        target_angle = self.goal_angle
        cv2.circle(image, (565, 305), 120, color, 4)
        cv2.putText(image, f'Goal     Angle : {target_angle}', (580, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, f'Cylinder Angle : {int(cylinder)}', (580, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, f'Episode : {str(episode)}', (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, f'Steps : {str(step)}', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

        cv2.imshow("State Image", image)
        cv2.waitKey(10)

    def calculate_reward(self, cylinder_angle):
        difference_cylinder_goal = np.abs(cylinder_angle - self.goal_angle)

        if difference_cylinder_goal <= 3:
            print("GOAL SUCCESS, REWARD = 1000")
            done = True
            reward_d = 1000
        else:
            done = False
            reward_d = -difference_cylinder_goal
        return reward_d, done
