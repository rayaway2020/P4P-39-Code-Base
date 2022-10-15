
import random

from vision_utilities_rotation import *
from motor_utilities_rotation import *


class RL_ENV:

    def __init__(self):

        self.cylinder_angle = 0

        self.position_joint_1_arm_1 = 0
        self.position_end_arm_1     = 0

        self.position_joint_1_arm_2 = 0
        self.position_end_arm_2 = 0

        self.goal_angle   = 0
        self.cylinder_angle = 0

        self.previos_cylinder_angle = 0


    def reset_env(self):
        # move the robot to home position:
        id_1_dxl_home_position = 390
        id_2_dxl_home_position = 350
        id_3_dxl_home_position = 685
        id_4_dxl_home_position = 655

        print("Sending Robot to Home Position")
        move_motor_step(id_1_dxl_home_position, id_2_dxl_home_position, id_3_dxl_home_position, id_4_dxl_home_position)

        #self.goal_angle = random.randint(5, 175)
        self.goal_angle = random.randint(-180, 180)
        print("New Goal angle generated", self.goal_angle)


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
        # generate the observation state
        # all  w.r.t the reference frame

        # See the readme file and draw to understand these values
        l_1 = 6.0  # size of link one, cm dimension
        l_2 = 4.5  # size of link two, cm dimension

        d_x_1 = 9.0  # x displacement of arm 1 wrt the reference frame
        d_x_2 = 2.5  # x displacement of arm 2 wrt the reference frame
        d_y = 4.0  # y displacement of arm 1 and arm2 wrt the reference frame

        tetha_1_arm_1, tetha_2_arm_1, tetha_1_arm_2, tetha_2_arm_2 = get_angles()

        x_joint_1_arm_1, y_joint_1_arm_1, x_end_arm1, y_end_arm1 = self.forward_kinematic(tetha_1_arm_1, tetha_2_arm_1,
                                                                                          l_1, l_2, d_x_1, d_y)

        x_joint_1_arm_2, y_joint_1_arm_2, x_end_arm2, y_end_arm2 = self.forward_kinematic(tetha_1_arm_2, tetha_2_arm_2,
                                                                                          l_1, l_2, d_x_2, d_y)

        self.position_joint_1_arm_1 = (x_joint_1_arm_1, y_joint_1_arm_1)
        self.position_end_arm_1 = (x_end_arm1, y_end_arm1)

        self.position_joint_1_arm_2 = (x_joint_1_arm_2, y_joint_1_arm_2)
        self.position_end_arm_2 = (x_end_arm2, y_end_arm2)

        self.cylinder_angle, image_to_display = calculate_cylinder_angle()


        if self.cylinder_angle is None:
            self.cylinder_angle = self.previos_cylinder_angle
        else:
            self.previos_cylinder_angle = self.cylinder_angle


        observation_space_n = self.position_joint_1_arm_1, self.position_end_arm_1, \
                            self.position_joint_1_arm_2, self.position_end_arm_2, \


        observation_space_n = [element for tupl in observation_space_n for element in tupl]  # Convert the tuples to

        observation_space_n.append(self.goal_angle)
        observation_space_n.append(self.cylinder_angle)

        return np.array(observation_space_n), image_to_display


    def env_step(self, actions):
        # put the outputs in operation step area

        id_1_dxl_goal_position = (actions[0] - (-1)) * (405 - 355) / (1 - (-1)) + 355
        id_2_dxl_goal_position = (actions[1] - (-1)) * (260 - 160) / (1 - (-1)) + 160
        id_3_dxl_goal_position = (actions[2] - (-1)) * (670 - 620) / (1 - (-1)) + 620
        id_4_dxl_goal_position = (actions[3] - (-1)) * (860 - 760) / (1 - (-1)) + 760

        id_1_dxl_goal_position = int(id_1_dxl_goal_position)
        id_2_dxl_goal_position = int(id_2_dxl_goal_position)
        id_3_dxl_goal_position = int(id_3_dxl_goal_position)
        id_4_dxl_goal_position = int(id_4_dxl_goal_position)

        #print("Motor Action Value:", id_1_dxl_goal_position, id_2_dxl_goal_position,
                                     #id_3_dxl_goal_position, id_4_dxl_goal_position)


        move_motor_step(id_1_dxl_goal_position, id_2_dxl_goal_position, id_3_dxl_goal_position, id_4_dxl_goal_position)


    def calculate_reward(self):

        difference_cylinder_goal = np.abs(self.cylinder_angle - self.goal_angle)

        print("Actual Position:", self.cylinder_angle, "Goal Angle:", self.goal_angle)

        if difference_cylinder_goal <= 3:
            print("GOAL SUCCESS, REWARD = 1000")
            done = True
            reward_d = 1000
        else:
            done = False
            reward_d = -difference_cylinder_goal

        return reward_d, done


    def close_env(self):
        motor_terminate()

