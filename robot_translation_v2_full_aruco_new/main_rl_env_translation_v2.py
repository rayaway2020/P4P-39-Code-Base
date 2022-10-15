from vision_utilities import Vision
from motor_utilities import Motor
import random
import cv2
import numpy as np

font = cv2.FONT_HERSHEY_SIMPLEX

class RL_ENV:

    def __init__(self):
        self.goal_angle = 0.0

        # -----> Previous config
        self.target_in_pixel_x = 200.0
        self.target_in_pixel_y = 100.0
        self.position_joint_1_arm_1 = 0
        self.position_end_arm_1     = 0

        self.position_joint_1_arm_2 = 0
        self.position_end_arm_2     = 0

        self.goal_position = 0
        self.cube_position = 0

        self.previos_goal = 0
        self.previos_cube = 0


        self.motors_config = Motor()
        self.vision_config = Vision()

    def reset_env(self):
        # move the robot to home position:
        id_1_dxl_home_position = 525
        id_2_dxl_home_position = 525
        id_3_dxl_home_position = 525
        id_4_dxl_home_position = 525

        print("Sending Robot to Home Position")
        self.motors_config.move_motor_step(id_1_dxl_home_position, id_2_dxl_home_position,
                                           id_3_dxl_home_position, id_4_dxl_home_position)

        # generate a new goal position value in PIXELS between the operation area
        #self.target_in_pixel_x = random.randint(150, 375)
        #self.target_in_pixel_y = random.randint(150, 220)
        #self.goal_one = [375, 210]
        #self.goal_two = [155, 215]

        # Choose a new goal position, two possible values
        flip = random.randint(0, 1)

        if flip == 0:
            self.target_in_pixel_x = random.randint(350.0, 420.0)
            self.target_in_pixel_y = random.randint(350.0, 410.0)
        else:
            self.target_in_pixel_x = random.randint(720.0, 800.0)
            self.target_in_pixel_y = random.randint(350.0, 410.0)
        # if flip ==0:
        #     self.target_in_pixel_x = 190
        #     self.target_in_pixel_y = 160
        # else:
        #     self.target_in_pixel_x = 415
        #     self.target_in_pixel_y = 160
        # if flip == 0:
        #     self.target_in_pixel_x = 314
        #     self.target_in_pixel_y = 100
        # else:
        #     self.target_in_pixel_x = 209
        #     self.target_in_pixel_y = 149


    def state_space_function(self):
        # while True:
        #     self.goal_position, image_to_display = self.vision_config.calculate_transformation_target(
        #         self.target_in_pixel_x,
        #         self.target_in_pixel_y)
        #     if image_to_display is not None:
        #         break
        while True:
            state_space_vector, raw_img, detection_status, cube_location, goal_location = self.vision_config.calculate_marker_pose(self.target_in_pixel_x, self.target_in_pixel_y)
            if detection_status:
                break
        self.goal_position = goal_location
        self.cube_position = cube_location
        return state_space_vector, raw_img

    def graphical_state_space_function(self):
        while True:
            observation_space, raw_img, detection_status, cube_location, goal_location = self.vision_config.calculate_marker_pose(self.target_in_pixel_x, self.target_in_pixel_y)
            if detection_status:
                break
        self.goal_position = goal_location
        self.cube_position = cube_location
        # and have all in the same style
        # observation_space = [element for tupl in observation_space for element in tupl]
        img_state = self.vision_config.plot_state_space(observation_space)
        return img_state, raw_img


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

    def env_step_discrete(self, actions):
        crash = self.motors_config.move_motor_step(actions[0], actions[1], actions[2], actions[3])
        if crash == 1:
            return crash

    def env_render(self, image, done, step):
        if done:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        text_in_target = (self.target_in_pixel_x - 15, self.target_in_pixel_y + 3)
        target = (self.target_in_pixel_x, self.target_in_pixel_y)
        cv2.circle(image, target, 25, color, -1)
        cv2.putText(image, 'Target', text_in_target, font, 0.3, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(step), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow("State Image", image)
        cv2.waitKey(10)

    def calculate_reward_continous(self, cylinder_angle):
        difference_cylinder_goal = np.abs(cylinder_angle - self.goal_angle)

        if difference_cylinder_goal <= 3:
            print("GOAL SUCCESS, REWARD = 1000")
            done = True
            reward_d = 1000
        else:
            done = False
            reward_d = -difference_cylinder_goal
        return reward_d, done


    def calculate_reward_discrete(self):

        distance_cube_goal = np.linalg.norm(self.cube_position - self.goal_position)

        if distance_cube_goal <= 1.0:
            print("GOAL SUCCESS, REWARD = 500")
            done = True
            reward_d = 500
        else:
            done = False
            reward_d = -distance_cube_goal
        return reward_d, done