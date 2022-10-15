import os
import cv2
import numpy as np
import math

matrix      = np.loadtxt("matrix.txt")
distortion  = np.loadtxt("distortion.txt")

# Aruco Dictionary
arucoDict   = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
arucoParams = cv2.aruco.DetectorParameters_create()

markerSizeInM = 0.025  # size of the aruco marker
cam = cv2.VideoCapture(0)  # open the camera

font = cv2.FONT_HERSHEY_SIMPLEX


def get_camera_image():
    ret, frame = cam.read()
    if ret:
        return frame


def calculate_cylinder_angle():
    image = get_camera_image()

    # Detect Aruco markers
    (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)

    if len(corners) > 0:  # if there are at least two ids on the screen

        try:
            # rotation and translation vector w.r.t camera of aruco markers i.e origen frame and cylinder
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, markerSizeInM, matrix, distortion)

            # all these values w.r.t camera frame
            mark_1 = tvec[0]  # reference frame marker
            mark_2 = tvec[1]  # cylinder frame marker

            mark_2_rvec = rvec[1]
            r_matrix, _ = cv2.Rodrigues(mark_2_rvec[0])
            psi, theta, phi = calculate_rotation_matrix(r_matrix)

            phi = math.degrees(phi)

            # coordinates w.r.t. reference marker
            #position_cylinder = mark_1 - mark_2  # this unit is meters
            #position_cylinder = position_cylinder * [-1, 1, 1]  # I need this for the x-axis to match with the reference frame
            #position_cube_cm = position_cylinder * 100

            return phi, image

        except:

            print("problem")
            return None, image

    else:
        print("problem cube marker not detected")
        return None, image


def calculate_rotation_matrix(R):
    '''
    From a paper by Gregory G. Slabaugh (undated),
    "Computing Euler angles from a rotation matrix
    '''
    phi = 0.0
    if isclose(R[2, 0], -1.0):
        theta = math.pi / 2.0
        psi = math.atan2(R[0, 1], R[0, 2])
    elif isclose(R[2, 0], 1.0):
        theta = -math.pi / 2.0
        psi = math.atan2(-R[0, 1], -R[0, 2])
    else:
        theta = -math.asin(R[2, 0])
        cos_theta = math.cos(theta)
        psi = math.atan2(R[2, 1] / cos_theta, R[2, 2] / cos_theta)
        phi = math.atan2(R[1, 0] / cos_theta, R[0, 0] / cos_theta)
    return psi, theta, phi


def isclose(x, y, rtol=1.e-5, atol=1.e-8):
    return abs(x-y) <= atol + rtol * abs(y)
