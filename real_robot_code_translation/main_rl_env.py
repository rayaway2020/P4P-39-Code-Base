"""
Author: David Valencia
Date: 03/ 05 /2022
Update: 10/ 05 /2022

Describer:
            Main Reinforcement Learning file for robot gripper test bed
            This script is the core of the project. The state-space, action, reward calculation and data from
            the camera and motors all are generated/read here

            state-space  = joint1 position(each arm),end effector position(each arm), goal position and cube position
            action-space = action vector size = 4 (one action for each motor)

            Here the robot status information is obtained from a combination of Aruco markers, robot forward kinematic,
            and real measures. An Aruco marker is required for this environment because it acts as a reference frame,
            i.e. all the distances are calculated w.r.t that marker.

            if close_env funct is called:
                1) DISABLE Torque for each motor
                2) Close the USB port
"""

import random
from motor_ultilities import *
from vision_utilities import *


class RL_ENV:

    def __init__(self):
        # self.target_in_pixel_x = 0
        # self.target_in_pixel_y = 0
        self.target_in_pixel_x = 200
        self.target_in_pixel_y = 100
        self.position_joint_1_arm_1 = 0
        self.position_end_arm_1     = 0

        self.position_joint_1_arm_2 = 0
        self.position_end_arm_2     = 0

        self.goal_position = 0
        self.cube_position = 0

        self.previos_goal = 0
        self.previos_cube = 0


    def reset_env(self):
        # ----> move the robot to home position:
        id_1_dxl_home_position = 500
        id_2_dxl_home_position = 500
        id_3_dxl_home_position = 500
        id_4_dxl_home_position = 500

        print("Sending Robot to Home Position")
        move_motor_step(id_1_dxl_home_position, id_2_dxl_home_position, id_3_dxl_home_position, id_4_dxl_home_position)

        print("New Goal point generated")

        # generate a new goal position value in PIXELS between the operation area
        #self.target_in_pixel_x = random.randint(150, 375)
        #self.target_in_pixel_y = random.randint(150, 220)
        #self.goal_one = [375, 210]
        #self.goal_two = [155, 215]

        # Choose a new goal position, two possible values
        flip = random.randint(0, 1)

        if flip == 0:
            self.target_in_pixel_x = random.randint(180, 220)
            self.target_in_pixel_y = random.randint(150, 200)
        else:
            self.target_in_pixel_x = random.randint(390, 440)
            self.target_in_pixel_y = random.randint(130, 190)

        # if flip == 1:
        #     self.target_in_pixel_x = 375
        #     self.target_in_pixel_y = 210
        # else:
        #     self.target_in_pixel_x = 155
        #     self.target_in_pixel_y = 215


    def generate_sample_act(self):
        act_m1 = np.clip(random.uniform(-1, 1), -1, 1)
        act_m2 = np.clip(random.uniform(-1, 1), -1, 1)
        act_m3 = np.clip(random.uniform(-1, 1), -1, 1)
        act_m4 = np.clip(random.uniform(-1, 1), -1, 1)
        action_vector = np.array([act_m1, act_m2, act_m3, act_m4])
        return action_vector


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
        # all  w.r.t the reference frame,
        # Numerical Vector value

        # See the readme file and draw to understand better these values
        l_1 = 6.0  # size of link one, cm dimension
        l_2 = 4.5  # size of link two, cm dimension

        d_x_1 = 9.0  # x displacement of arm 1 wrt the reference frame
        d_x_2 = 2.5  # x displacement of arm 2 wrt the reference frame
        d_y   = 4.0  # y displacement of arm 1 and arm2 wrt the reference frame

        tetha_1_arm_1, tetha_2_arm_1, tetha_1_arm_2, tetha_2_arm_2 = get_angles()

        x_joint_1_arm_1, y_joint_1_arm_1, x_end_arm1, y_end_arm1 = self.forward_kinematic(tetha_1_arm_1, tetha_2_arm_1,
                                                                                          l_1, l_2, d_x_1, d_y)

        x_joint_1_arm_2, y_joint_1_arm_2, x_end_arm2, y_end_arm2 = self.forward_kinematic(tetha_1_arm_2, tetha_2_arm_2,
                                                                                          l_1, l_2, d_x_2, d_y)

        self.position_joint_1_arm_1 = (x_joint_1_arm_1, y_joint_1_arm_1)
        self.position_end_arm_1     = (x_end_arm1, y_end_arm1)

        self.position_joint_1_arm_2 = (x_joint_1_arm_2, y_joint_1_arm_2)
        self.position_end_arm_2     = (x_end_arm2, y_end_arm2)

        self.goal_position, image_to_display = calculate_transformation_target(self.target_in_pixel_x,
                                                                               self.target_in_pixel_y)

        while True:
            self.cube_position, found = calculate_cube_position()
            if found:
                break
            else:
                print("Waiting for markers to be detected")

        # THIS IS CASE THAT THERE ARE NO MARKS DETECTED
        if self.goal_position is None:
            print("problem reference marker")
            self.goal_position = self.previos_goal
        else:
            self.previos_goal  = self.goal_position

        if self.cube_position is None:
            print("--------------This step cant really find the cube------------")
            self.cube_position = self.previos_cube
        else:
            self.previos_cube  = self.cube_position


        observation_space = self.position_joint_1_arm_1, self.position_end_arm_1, \
                            self.position_joint_1_arm_2, self.position_end_arm_2, \
                            tuple(self.goal_position), tuple(self.cube_position)

        # to have all in the same style
        # Then convert the tuples to one-dimensional list

        observation_space = [element for tupl in observation_space for element in tupl]
        return np.array(observation_space), image_to_display


    def graphical_state_space_funct(self):
        # generate the observation state as IMAGE including
        # all  w.r.t the reference frame

        # See the readme file and draw to understand these values
        l_1 = 6.0  # size of link one, cm dimension
        l_2 = 4.5  # size of link two, cm dimension

        d_x_1 = 9.0  # x displacement of arm 1 wrt the reference frame
        d_x_2 = 2.5  # x displacement of arm 2 wrt the reference frame
        d_y   = 4.0  # y displacement of arm 1 and arm2 wrt the reference frame

        tetha_1_arm_1, tetha_2_arm_1, tetha_1_arm_2, tetha_2_arm_2 = get_angles()

        x_joint_1_arm_1, y_joint_1_arm_1, x_end_arm1, y_end_arm1 = self.forward_kinematic(tetha_1_arm_1, tetha_2_arm_1,
                                                                                          l_1, l_2, d_x_1, d_y)

        x_joint_1_arm_2, y_joint_1_arm_2, x_end_arm2, y_end_arm2 = self.forward_kinematic(tetha_1_arm_2, tetha_2_arm_2,
                                                                                          l_1, l_2, d_x_2, d_y)

        self.position_joint_1_arm_1 = (x_joint_1_arm_1, y_joint_1_arm_1)
        self.position_end_arm_1 = (x_end_arm1, y_end_arm1)

        self.position_joint_1_arm_2 = (x_joint_1_arm_2, y_joint_1_arm_2)
        self.position_end_arm_2 = (x_end_arm2, y_end_arm2)

        self.goal_position, image_to_display = calculate_transformation_target(self.target_in_pixel_x,
                                                                               self.target_in_pixel_y)

        self.cube_position = calculate_cube_position()

        # THIS IS CASE THAT THERE ARE NO MARKS DETECTED
        if self.goal_position is None:
            print("problem reference marker")
            self.goal_position = self.previos_goal
        else:
            self.previos_goal  = self.goal_position

        if self.cube_position is None:
            self.cube_position = self.previos_cube
        else:
            self.previos_cube  = self.cube_position


        self.position_joint_1_arm_1_n = self.normalizar_obsevation(self.position_joint_1_arm_1)
        self.position_joint_1_arm_2_n = self.normalizar_obsevation(self.position_joint_1_arm_2)
        self.position_end_arm_1_n     = self.normalizar_obsevation(self.position_end_arm_1)
        self.position_end_arm_2_n     = self.normalizar_obsevation(self.position_end_arm_2)
        self.cube_position_n          = self.normalizar_obsevation(self.cube_position)
        self.goal_position_n          = self.normalizar_obsevation(self.goal_position)


        observation_space = self.position_joint_1_arm_1_n, self.position_end_arm_1_n, \
                            self.position_joint_1_arm_2_n, self.position_end_arm_2_n, \
                            self.goal_position_n, self.cube_position_n

        # and have all in the same style
        observation_space = [element for tupl in observation_space for element in tupl]
        img_state = plot_state_space(observation_space)

        return img_state


    def normalizar_obsevation(self, data):

        x_value = (data[0] + 25) * 5
        y_value = (data[1] + 25) * 5

        return x_value, y_value



    def env_step(self, actions):
        # put the outputs in operation step area

        id_1_dxl_goal_position = (actions[0] - (-1)) * (700 - 300) / (1 - (-1)) + 300
        id_2_dxl_goal_position = (actions[1] - (-1)) * (700 - 300) / (1 - (-1)) + 300
        id_3_dxl_goal_position = (actions[2] - (-1)) * (700 - 300) / (1 - (-1)) + 300
        id_4_dxl_goal_position = (actions[3] - (-1)) * (700 - 300) / (1 - (-1)) + 300

        id_1_dxl_goal_position = int(id_1_dxl_goal_position)
        id_2_dxl_goal_position = int(id_2_dxl_goal_position)
        id_3_dxl_goal_position = int(id_3_dxl_goal_position)
        id_4_dxl_goal_position = int(id_4_dxl_goal_position)

        #print("Motor Action Value:", id_1_dxl_goal_position, id_2_dxl_goal_position,
                                     #id_3_dxl_goal_position, id_4_dxl_goal_position)

        crash = move_motor_step(id_1_dxl_goal_position, id_2_dxl_goal_position, id_3_dxl_goal_position,
                                id_4_dxl_goal_position)
        if crash == 1:
            return crash


    def env_step_discrete(self, actions):
        crash = move_motor_step(actions[0], actions[1], actions[2], actions[3])
        if crash == 1:
            return crash

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
        cv2.imshow("State Image", image)
        cv2.waitKey(10)



    def calculate_reward_continuous(self):

        #goal_position = np.array(self.goal_position)
        #cube          = np.array(self.cube_position)
        #distance_cube_goal = np.linalg.norm(cube - goal_position)

        distance_cube_goal = np.linalg.norm(self.cube_position - self.goal_position)

        print("Distance", distance_cube_goal)

        if distance_cube_goal <= 1.50:
            print("GOAL SUCCESS, REWARD = 100")
            done = True
            reward_d = 100
        else:
            done = False
            reward_d = -distance_cube_goal
        return reward_d, done


    def calculate_reward_discrete(self):

        #goal_position = np.array(self.goal_position)
        #cube          = np.array(self.cube_position)
        #distance_cube_goal = np.linalg.norm(cube - goal_position)

        distance_cube_goal = np.linalg.norm(self.cube_position - self.goal_position)

        print("Distance", distance_cube_goal)

        if distance_cube_goal <= 1.50:
            print("GOAL SUCCESS, REWARD = 100")
            done = True
            reward_d = 500
        else:
            done = False
            reward_d = -distance_cube_goal
        return reward_d, done


    def close_env(self):
        motor_terminate()
