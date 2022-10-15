
import cv2
import numpy as np

# From calibration matrix
matrix     = np.loadtxt(open('matrix.txt', 'rb'))
distortion = np.loadtxt(open("distortion.txt", "rb"))

arucoDict  = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
#arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)

arucoParams = cv2.aruco.DetectorParameters_create()

# Target Point in Screen (Pixel Values)
target_in_pixel_x = 155
target_in_pixel_y = 215

markerSizeInM = 0.025 

cam = cv2.VideoCapture(0)

while cam.isOpened():
    ret, image = cam.read()

    if ret:
        # Detect Aruco markers
        (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)
        cv2.aruco.drawDetectedMarkers(image, corners, borderColor=(0, 0, 255))
        cv2.circle(image, (target_in_pixel_x, target_in_pixel_y), 15, (0, 0, 250), -1) # just plot a circle for target point
        
        if len(corners) > 0:
            s = np.abs(corners[0][0][0][0] - corners[0][0][3][0])  # dimension side of the square
            s = s/2 
            a = np.array([[-1, 1], [1, 1], [1, -1], [-1, -1]])  # support matrix to create the fake corners
            
            # virtually create corners in the target point as a "fake marker"
            target_marker_corners = [a * s + (target_in_pixel_x, target_in_pixel_y)]
            target_marker_corners = [np.array(target_marker_corners, dtype="float32")]

            # rotation and translation of virtual target marker w.r.t camera
            target_rvec , target_tvec, _ = cv2.aruco.estimatePoseSingleMarkers(target_marker_corners, markerSizeInM, matrix, distortion)
            cv2.aruco.drawDetectedMarkers(image, target_marker_corners, borderColor=(0, 0, 255))

            # rotation and translation of reference marker w.r.t camera
            reference_rvec, refence_tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, markerSizeInM, matrix, distortion)

            mark_1 = refence_tvec[0]  # reference frame marker
            target = target_tvec     # target virtual marker

            position_target = mark_1 - target  # this unit is meters
            position_target = position_target * [-1, 1, 1]
            position_target_cm = position_target * 100

            print("Position Fake target Point", position_target_cm)

            # another way to do the same
            '''
            rvec , tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, markerSizeInM, matrix, distortion) # rotation and traslation w.r.t camera of aruco markers i.e origen
            
            z = tvec[0,0,2]

            cx = matrix[0,2]
            fx = matrix[0,0]
            cy = matrix[1,2]
            fy = matrix[1,1]

            px = (target_in_pixel_x - cx) / fx
            py = (target_in_pixel_y - cy) / fy
         
            px = px * z
            py = py * z
            pz = z

            #coordinates w.r.t. camera frame
            reference_marker_wrt_camera = tvec[0] # reference frame marker
            target_point_wrt_camera     = (px, py, pz)

            # coordinates w.r.t. reference marker
            position_target = reference_marker_wrt_camera - target_point_wrt_camera  # this unit is meters
            position_target = position_target * [-1, 1, 1]
            position_target_cm = position_target * 100
            print(position_target_cm)
            '''

        else:
            print("need at least one marker")

    cv2.imshow('Main Image', image)
    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
