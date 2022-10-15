"""
Author: David Valencia
Date: 10/ 05 /2022

Description:
             functions associated with the camera and opencv to obtain the position of aruco markers,
             make transformations and calculate the distance
"""

import os
import cv2
import numpy as np

from os.path import expanduser
home = expanduser("~")

# These values come after calibrating the camera
matrix_dist_path = os.path.join(home, 'workspace/test_bed_RL_robot/camera_calibration_files')

matrix      = np.loadtxt(open(os.path.join(matrix_dist_path, "matrix.txt"), 'rb'))
distortion  = np.loadtxt(open(os.path.join(matrix_dist_path, "distortion.txt"), "rb"))

# Aruco Dictionary
arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)

arucoParams   = cv2.aruco.DetectorParameters_create()

markerSizeInM = 0.025  # size of the aruco marker
cam = cv2.VideoCapture(0)  # open the camera

font = cv2.FONT_HERSHEY_SIMPLEX


def get_camera_image():
    ret, frame = cam.read()
    if ret:
        return frame


def calculate_transformation_target(target_x_pix, target_y_pix):

    image = get_camera_image()

    # Detect Aruco markers
    (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)
    #cv2.aruco.drawDetectedMarkers(image, corners, borderColor=(0, 0, 255))

    if len(corners) > 0:  # if there is at least one id on the screen

        # calculate the size of the aruco mark
        s = np.abs(corners[0][0][0][0] - corners[0][0][3][0])  # dimension side of the square
        s = s / 2
        a = np.array([[-1, 1], [1, 1], [1, -1], [-1, -1]])  # support matrix to create the fake corners

        # virtually create corners around the target point as a "fake marker"
        target_marker_corners = [a * s + (target_x_pix, target_y_pix)]
        target_marker_corners = [np.array(target_marker_corners, dtype="float32")]

        # rotation and translation of virtual target marker w.r.t camera
        target_rvec, target_tvec, _ = cv2.aruco.estimatePoseSingleMarkers(target_marker_corners, markerSizeInM,
                                                                          matrix, distortion)

        # rotation and translation of real markers w.r.t camera
        reference_rvec, refence_tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, markerSizeInM,
                                                                              matrix, distortion)

        # all these values w.r.t camera frame
        mark_1 = refence_tvec[0]  # reference frame marker
        target = target_tvec      # target virtual marker

        # coordinates w.r.t. reference marker
        position_target = mark_1 - target  # this unit is meters
        position_target = position_target * [-1, 1, 1]  # I need this for the x-axis to match with the reference frame
        position_target_cm = position_target * 100  # this unit is CM


        # another way to do the same calculation
        '''
        rvec , tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, markerSizeInM, matrix, distortion) # rotation and traslation w.r.t camera of aruco markers i.e origen
        
        z = tvec[0,0,2]

        cx = matrix[0,2]
        fx = matrix[0,0]
        cy = matrix[1,2]
        fy = matrix[1,1]

        px = (target_x_pix - cx) / fx
        py = (target_y_pix - cy) / fy
        
        px = px * z
        py = py * z
        pz = z

        # coordinates w.r.t. camera frame
        reference_marker_wrt_camera = tvec[0] # reference frame marker
        target_point_wrt_camera     = (px, py, pz)

        # coordinates w.r.t. reference marker
        position_target = reference_marker_wrt_camera - target_point_wrt_camera  # this unit is meters
        position_target_cm = position_target * 100
        print(position_target_cm)
        '''

        return position_target_cm[0][0][:-1], image  # no# I am not considering z axis here
    else:
        print("problem target or references")


def calculate_cube_position():

    image = get_camera_image()

    # Detect Aruco markers
    (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)

    if len(corners) > 1:  # if there are at least two ids on the screen

        # rotation and translation w.r.t camera of aruco markers i.e. origen frame and cube
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, markerSizeInM, matrix, distortion)

        # all these values w.r.t camera frame
        mark_1 = tvec[0]  # reference frame marker
        mark_2 = tvec[1]  # cube frame marker

        # coordinates w.r.t. reference marker
        position_cube = mark_1 - mark_2  # this unit is meters
        position_cube = position_cube * [-1, 1, 1]  # I need this for the x-axis to match with the reference frame
        position_cube_cm = position_cube * 100

        return position_cube_cm[0][:-1], True  # I am not considering z axis here

    else:
        print("problem cube marker")
        return 0, False


def plot_state_space(observation_space):

    image_height = 300
    image_width  = 300
    size = 10
    cube_targ_size = 10

    number_of_color_channels = 3
    color = (202, 202, 202)

    img = np.full((image_height, image_width, number_of_color_channels), color, dtype=np.uint8)

    joint_1_arm_1 = (int(observation_space[0]), int(observation_space[1]))
    end_arm_1     = (int(observation_space[2]), int(observation_space[3]))
    joint_1_arm_2 = (int(observation_space[4]), int(observation_space[5]))
    end_arm_2     = (int(observation_space[6]), int(observation_space[7]))
    target        = (int(observation_space[8]), int(observation_space[9]))
    cube          = (int(observation_space[10]), int(observation_space[11]))


    cv2.rectangle(img, (cube[0], cube[1]), (cube[0] + cube_targ_size, cube[1] + cube_targ_size), (0, 255, 0), -1)
    cv2.rectangle(img, (target[0], target[1]), (target[0] + cube_targ_size, target[1] + cube_targ_size), (0, 0, 255), -1)

    cv2.line(img, (joint_1_arm_1[0]+5, joint_1_arm_1[1]+5), (end_arm_1[0]+5, end_arm_1[1]+5), (0, 0, 0), 2)
    cv2.line(img, (joint_1_arm_2[0]+5, joint_1_arm_2[1]+5), (end_arm_2[0]+5, end_arm_2[1] + 5), (0, 0, 0), 2)

    cv2.rectangle(img, (joint_1_arm_1[0], joint_1_arm_1[1]), (joint_1_arm_1[0] + size, joint_1_arm_1[1] + size), (0, 0, 153), -1)
    cv2.rectangle(img, (end_arm_1[0], end_arm_1[1]), (end_arm_1[0] + size, end_arm_1[1] + size), (204, 102, 0), -1)

    cv2.rectangle(img, (joint_1_arm_2[0], joint_1_arm_2[1]), (joint_1_arm_2[0] + size, joint_1_arm_2[1] + size), (128, 128, 255), -1)
    cv2.rectangle(img, (end_arm_2[0], end_arm_2[1]), (end_arm_2[0] + size, end_arm_2[1] + size), (128, 0, 128), -1)

    return img



def close_camera_set():
    cam.release()
    cv2.destroyAllWindows()


