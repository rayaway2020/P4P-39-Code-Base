import cv2
import random
import numpy as np
from motor_utilities import Motor
from vision_utilities import Vision


class RL_ENV:
    def __init__(self, usb_index='/dev/ttyUSB0', camera_index=0):

        self.motors_config = Motor(usb_index)
        self.vision_config = Vision(camera_index)

        self.target_in_pixel_x = 500
        self.target_in_pixel_y = 370

        self.cube_position = 0
        self.target_position = 0

        self.counter_success = 0

    def reset_env(self):

        # move the robot to home position:
        id_1_dxl_home_position = 510
        id_2_dxl_home_position = 472
        id_3_dxl_home_position = 492
        id_4_dxl_home_position = 510

        print("Sending Robot to Home Position")
        self.motors_config.move_motor_step(id_1_dxl_home_position, id_2_dxl_home_position,
                                           id_3_dxl_home_position, id_4_dxl_home_position)

        print("Generating New Target Position")
        target_index = random.randint(0, 1)
        if target_index == 1:
            self.target_in_pixel_x = random.randint(830, 870)
            self.target_in_pixel_y = random.randint(370, 430)
        else:
            self.target_in_pixel_x = random.randint(450, 500)
            self.target_in_pixel_y = random.randint(370, 440)

        '''
        print("Generating New Target Position")
        # Four possible target points:
        target_index = random.randint(1, 4)

        if target_index == 1:
            self.target_in_pixel_x = 870
            self.target_in_pixel_y = 430

        elif target_index == 2:
            self.target_in_pixel_x = 830
            self.target_in_pixel_y = 370

        elif target_index == 3:
            self.target_in_pixel_x = 450
            self.target_in_pixel_y = 440
        elif target_index == 4:
            self.target_in_pixel_x = 500
            self.target_in_pixel_y = 370
        '''

    def generate_sample_act(self):
        act_m1 = np.clip(random.uniform(-1, 1), -1, 1)
        act_m2 = np.clip(random.uniform(-1, 1), -1, 1)
        act_m3 = np.clip(random.uniform(-1, 1), -1, 1)
        act_m4 = np.clip(random.uniform(-1, 1), -1, 1)
        action_vector = np.array([act_m1, act_m2, act_m3, act_m4])
        return action_vector

    def state_space_function(self):
        while True:
            state_space_vector, raw_img, detection_status, _, _ = self.vision_config.calculate_marker_pose(
                self.target_in_pixel_x, self.target_in_pixel_y)
            if detection_status:
                self.target_position = state_space_vector[-2:]
                self.cube_position = state_space_vector[-4:-2]
                break
            else:
                print("waiting for camera and marks")
        return np.array(state_space_vector), raw_img

    def state_space_function_discrete(self):
        while True:
            state_space_vector, raw_img, detection_status, cube_location, goal_location = self.vision_config.calculate_marker_pose(
                self.target_in_pixel_x, self.target_in_pixel_y)
            # result = self.thread_function()
            # print(result)
            if detection_status:
                break
        self.goal_position = goal_location
        self.cube_position = cube_location
        return state_space_vector, raw_img

    def graphical_state_space_function(self):
        # generate the observation state as IMAGE
        while True:
            state_space_vector, raw_img, detection_status = self.vision_config.calculate_marker_pose(
                self.target_in_pixel_x,
                self.target_in_pixel_y)
            if detection_status:
                state_space_vector = [element for state_space_list in state_space_vector for element in
                                      state_space_list]
                self.target_position = state_space_vector[-2:]
                self.cube_position = state_space_vector[-4:-2]
                img_state_representation = self.vision_config.plot_state_space(state_space_vector)
                break
            else:
                print("waiting for camera and marks")
        return img_state_representation, raw_img

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

        self.motors_config.move_motor_step(id_1_dxl_goal_position,
                                           id_2_dxl_goal_position,
                                           id_3_dxl_goal_position,
                                           id_4_dxl_goal_position)

    def env_render(self, image, done=False, step=1, episode=1, mode="Exploration"):
        if done:
            if mode == "Exploration":
                color = (0, 255, 0)
            else:
                self.counter_success += 1
                color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        target = (self.target_in_pixel_x, self.target_in_pixel_y)
        text_in_target = (self.target_in_pixel_x - 15, self.target_in_pixel_y + 3)
        cv2.circle(image, target, 18, color, -1)
        cv2.putText(image, 'Target', text_in_target, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, f'Episode : {str(episode)}', (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(image, f'Steps : {str(step)}', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, f'Success Counter : {str(self.counter_success)}', (400, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, f'Stage : {mode}', (900, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow("State Image", image)
        cv2.waitKey(10)

    def calculate_reward(self):

        cube_position = np.array(self.cube_position)
        target_position = np.array(self.target_position)

        distance_cube_goal = np.linalg.norm(cube_position - target_position)  # millimeters distance
        # print("Distance to goal:", distance_cube_goal, "mm")

        if distance_cube_goal <= 10:  # millimeters
            print("GOAL SUCCESS, REWARD = 500")
            done = True
            reward_d = np.float64(500)
        else:
            done = False
            reward_d = -distance_cube_goal

        return reward_d, done


font = cv2.FONT_HERSHEY_SIMPLEX


class RL_ENV_Dis:

    def __init__(self, usb_index='/dev/ttyUSB0', camera_index=0):
        self.goal_angle = 0.0

        # -----> Previous config
        self.target_in_pixel_x = 200.0
        self.target_in_pixel_y = 100.0
        self.position_joint_1_arm_1 = 0
        self.position_end_arm_1 = 0

        self.position_joint_1_arm_2 = 0
        self.position_end_arm_2 = 0

        self.goal_position = 0
        self.cube_position = 0

        self.previos_goal = 0
        self.previos_cube = 0

        self.motors_config = Motor(usb_index)
        self.vision_config = Vision(camera_index)

    def reset_env(self):
        # move the robot to home position:
        id_1_dxl_home_position = 525
        id_2_dxl_home_position = 525
        id_3_dxl_home_position = 525
        id_4_dxl_home_position = 525

        print("Sending Robot to Home Position")
        self.motors_config.move_motor_step(id_1_dxl_home_position, id_2_dxl_home_position,
                                           id_3_dxl_home_position, id_4_dxl_home_position)
        # Choose a new goal position, two possible values
        flip = random.randint(0, 1)
        if flip == 0:
            self.target_in_pixel_x = random.randint(360.0, 480.0)
            self.target_in_pixel_y = random.randint(300.0, 400.0)
        else:
            self.target_in_pixel_x = random.randint(740.0, 840.0)
            self.target_in_pixel_y = random.randint(300.0, 400.0)

    def state_space_function(self):
        while True:
            state_space_vector, raw_img, detection_status, cube_location, goal_location = self.vision_config.calculate_marker_pose(
                self.target_in_pixel_x, self.target_in_pixel_y)
            if detection_status:
                break
        self.goal_position = goal_location
        self.cube_position = cube_location
        return state_space_vector, raw_img

    def graphical_state_space_function(self):
        while True:
            observation_space, raw_img, detection_status, cube_location, goal_location = self.vision_config.calculate_marker_pose(
                self.target_in_pixel_x, self.target_in_pixel_y)
            if detection_status:
                break
        self.goal_position = goal_location
        self.cube_position = cube_location
        img_state = self.vision_config.plot_state_space(observation_space)
        return img_state, raw_img

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

    def calculate_reward_discrete(self):

        distance_cube_goal = np.linalg.norm(self.cube_position - self.goal_position)

        if distance_cube_goal <= 10:
            print("GOAL SUCCESS, REWARD = 500")
            done = True
            reward_d = 500
        else:
            done = False
            reward_d = -distance_cube_goal
        return reward_d, done
