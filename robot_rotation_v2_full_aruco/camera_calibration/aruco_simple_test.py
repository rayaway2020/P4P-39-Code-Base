

import cv2
import numpy as np

# From calibration matrix
matrix     = np.loadtxt(open("matrix.txt", "rb"))
distortion = np.loadtxt(open("distortion.txt", "rb"))


arucoDict 	= cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
arucoParams = cv2.aruco.DetectorParameters_create()
cam         = cv2.VideoCapture(0)

while cam.isOpened():

    ret, image = cam.read()

    if ret:
        # Detect Aruco markers
        (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)
        print (ids)
        # Plot Aruco markers
        cv2.aruco.drawDetectedMarkers(image, corners, borderColor=(0, 255, 255))
        markerSizeInMillimeter = 18
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, markerSizeInMillimeter, matrix, distortion)


    cv2.imshow('Camera Frame', image)
    if cv2.waitKey(1) == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
