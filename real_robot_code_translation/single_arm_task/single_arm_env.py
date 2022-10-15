
import random
import numpy as np
from single_arm_motor_utility import *
from vision_utilities import *


class SingleArmEnv:
    def __init__(self):
        self.target_in_pixel_x = 300
        self.target_in_pixel_y = 140

        self.position_joint_1_arm_1 = 0
        self.position_end_arm_1 = 0

        self.goal_position = 0
        self.previos_goal  = 0

    def reset_env(self):
        # move the robot to home position:
        id_1_dxl_home_position = 511 #310
        id_2_dxl_home_position = 511 #310

        print("Sending Arm to Home Position")
        move_motor_step(id_1_dxl_home_position, id_2_dxl_home_position)

        print("New Goal point generated")
        self.target_in_pixel_x = random.randint(150, 500)  # todo check this range
        self.target_in_pixel_y = random.randint(180, 205)  # todo check this range

        #self.goal_position = np.array([random.randint(0, 10), random.randint(10, 12)])


    def forward_kinematic(self, tetha1, tetha2, l1, l2, d_x, d_y):

        tetha1_rad = np.deg2rad(tetha1)
        tetha2_rad = np.deg2rad(tetha2)

        x_1 = l1 * np.cos(tetha1_rad)
        y_1 = l1 * np.sin(tetha1_rad)

        x_2 = x_1 + l2 * np.cos(tetha1_rad + tetha2_rad)
        y_2 = y_1 + l2 * np.sin(tetha1_rad + tetha2_rad)

        # joint 1 position w.r.t the reference frame
        x_1_r = x_1 + d_x
        y_1_r = y_1 + d_y

        # end-effector position w.r.t the reference frame
        x_2_r = d_x + x_2
        y_2_r = d_y + y_2

        return x_1_r, y_1_r, x_2_r, y_2_r


    def state_space_funct(self):

        l_1 = 6.0  # size of link one, cm dimension
        l_2 = 4.5  # size of link two, cm dimension

        d_x_1 = 9.0  # x displacement of arm 1 wrt the reference frame
        d_y   = 4.0  # y displacement of arm 1 and arm2 wrt the refere  estaba 4.5

        tetha_1_arm_1, tetha_2_arm_1 = get_angles()
        x_joint_1_arm_1, y_joint_1_arm_1, x_end_arm1, y_end_arm1 = self.forward_kinematic(tetha_1_arm_1, tetha_2_arm_1,
                                                                                          l_1, l_2, d_x_1, d_y)

        self.position_joint_1_arm_1 = (x_joint_1_arm_1, y_joint_1_arm_1)
        self.position_end_arm_1     = (x_end_arm1, y_end_arm1)

        self.goal_position, image_to_display = calculate_transformation_target(self.target_in_pixel_x, self.target_in_pixel_y)

        # THIS IS CASE THAT THERE ARE NO MARKS DETECTED
        if self.goal_position is None:
            print("problem reference marker")
            self.goal_position = self.previos_goal
        else:
            self.previos_goal  = self.goal_position


        observation_space = self.position_joint_1_arm_1, self.position_end_arm_1, tuple(self.goal_position)

        observation_space = [element for tupl in observation_space for element in tupl]

        return np.array(observation_space), image_to_display


    def env_step(self, actions):

        # put the outputs in operation step area
        id_1_dxl_goal_position = (actions[0] - (-1)) * (700 - 300) / (1 - (-1)) + 300
        id_2_dxl_goal_position = (actions[1] - (-1)) * (700 - 300) / (1 - (-1)) + 300

        id_1_dxl_goal_position = int(id_1_dxl_goal_position)
        id_2_dxl_goal_position = int(id_2_dxl_goal_position)

        move_motor_step(id_1_dxl_goal_position, id_2_dxl_goal_position)

    def env_render(self, image, done, step):

        if done:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        text_in_target = (self.target_in_pixel_x - 15, self.target_in_pixel_y + 3)
        target = (self.target_in_pixel_x, self.target_in_pixel_y)
        cv2.circle(image, target, 18, color, -1)
        cv2.putText(image, 'Target', text_in_target, font, 0.3, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(step), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow("State", image)
        cv2.waitKey(10)

    def calculate_reward(self):

        dist_target_end = np.linalg.norm(self.position_end_arm_1 - self.goal_position)
        print("Distance:", dist_target_end)

        if dist_target_end <= 1.5:
            print("GOAL SUCCESS, REWARD = 100")
            done = True
            reward_d = 100
        else:
            done = False
            reward_d = -dist_target_end
        return reward_d, done



