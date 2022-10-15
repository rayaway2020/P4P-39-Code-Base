
import cv2
import numpy as np

# From calibration matrix
matrix     = np.loadtxt(open("matrix.txt", "rb"))
distortion = np.loadtxt(open("distortion.txt", "rb"))

arucoDict 	= cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
#arucoDict 	= cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)

arucoParams = cv2.aruco.DetectorParameters_create()

cam = cv2.VideoCapture(3)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

while cam.isOpened():
	ret, image = cam.read()
	if ret:
		# Detect Aruco markers
		(corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)
		cv2.aruco.drawDetectedMarkers(image, corners, ids, borderColor=(0, 0, 255))

		markerSizeInM = 0.018
		rvec , tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, markerSizeInM, matrix, distortion)

		if len(corners) > 1:  # meaning that there are at least two markers

			mark_1 = tvec[0]
			mark_2 = tvec[1]

			position_cube = mark_1 - mark_2  # this unit is meters
			position_cube = position_cube * [-1, 1, 1]  # I need this for the x-axis to match with the reference frame
			position_cube_cm = position_cube * 100  # (x, y)

			dist = np.linalg.norm(mark_1 - mark_2)
			#dist = np.linalg.norm(mark_1[0][:-1] - mark_2[0][:-1])
			
			print("Distance between markers:", dist, "M")
		
		else:
			print("Need more markers")

	# cv2.namedWindow("output", cv2.WINDOW_AUTOSIZE)
	# Naming a window
	cv2.namedWindow("Resized_Window", cv2.WINDOW_AUTOSIZE)

	# Using resizeWindow()
	# cv2.resizeWindow("Resized_Window", 1080, 720)
	# imS = cv2.resize(image, (1080, 720))
	cv2.imshow('Resized_Window', image)



	if cv2.waitKey(1) == ord('q'):
		break


cam.release()
cv2.destroyAllWindows()
