import cv2
import numpy as np

distortion = np.loadtxt("../camera_calibration_files/distortion.txt")
matrix = np.loadtxt("../camera_calibration_files/matrix.txt")


class Vision:

    def __init__(self, camera_index=2):

        # Aruco Dictionary
        self.arucoDict   = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        self.arucoParams = cv2.aruco.DetectorParameters_create()

        self.markerSize = 18  # size of the aruco marker millimeters
        self.camera     = cv2.VideoCapture(camera_index)  # open the camera
        self.robot_marks_id = [0, 1, 2, 3, 4, 5, 6]  # the id for each market in the robot

        self.vision_flag_status = False

    def get_camera_image(self):
        ret, frame = self.camera.read()
        if ret:
            return frame

    def get_index_id(self, aruco_id, id_detected):
        for id_index, id in enumerate(id_detected):
            if id == aruco_id:
                return id_index
        return -1


    def calculate_target_position(self, tvec, target_x, target_y):
        z  = tvec[0, 0, 2]
        cx = matrix[0, 2]
        fx = matrix[0, 0]
        cy = matrix[1, 2]
        fy = matrix[1, 1]

        px = (target_x - cx) / fx
        py = (target_y - cy) / fy

        px = px * z
        py = py * z
        #pz = z

        target_point_wrt_camera = (px, py)  # I am not considering z axis here
        return np.array(target_point_wrt_camera)


    def calculate_marker_pose(self, target_x_pix=600, target_y_pix=400):
        # get image from camera
        image = self.get_camera_image()

        # Detect Aruco markers, corners and IDs
        (corners, IDs, rejected) = cv2.aruco.detectMarkers(image, self.arucoDict, parameters=self.arucoParams)
        cv2.aruco.drawDetectedMarkers(image, corners, borderColor=(0, 0, 255))

        # rotation and translation vector w.r.t camera
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.markerSize, matrix, distortion)

        try:
            if len(IDs) >= 7:
                # Get the right index for each id of the detected markers
                for id_marker in self.robot_marks_id:

                    index = self.get_index_id(id_marker, IDs)

                    if id_marker == 0:
                        joint_0_arm_1_marker_index = index

                    if id_marker == 1:
                        joint_0_arm_2_marker_index = index

                    if id_marker == 2:
                        joint_1_arm_1_marker_index = index

                    if id_marker == 3:
                        joint_1_arm_2_marker_index = index

                    if id_marker == 4:
                        end_arm_1_marker_index = index

                    if id_marker == 5:
                        end_arm_2_marker_index = index

                    if id_marker == 6:
                        cube_marker_index = index


                joint_0_arm_1_location = tvec[joint_0_arm_1_marker_index][0][:-1]  # I am not considering z axis here
                joint_0_arm_2_location = tvec[joint_0_arm_2_marker_index][0][:-1]

                joint_1_arm_1_location = tvec[joint_1_arm_1_marker_index][0][:-1]
                joint_1_arm_2_location = tvec[joint_1_arm_2_marker_index][0][:-1]

                end_arm_1_location = tvec[end_arm_1_marker_index][0][:-1]
                end_arm_2_location = tvec[end_arm_2_marker_index][0][:-1]

                cube_location = tvec[cube_marker_index][0][:-1]

                target_location = self.calculate_target_position(tvec, target_x_pix, target_y_pix)

                state_space = (joint_0_arm_1_location, joint_0_arm_2_location,
                               joint_1_arm_1_location, joint_1_arm_2_location,
                               end_arm_1_location, end_arm_2_location,
                               cube_location, target_location)

                self.vision_flag_status = True
                return state_space, image, self.vision_flag_status

            else:
                print("not all aruco markers detected")
                self.vision_flag_status = False
                return 0, image, self.vision_flag_status

        except:
            print("camara obstruction")
            self.vision_flag_status = False
            return 0, image, self.vision_flag_status


    def plot_state_space(self, observation_space_vector):

        image_height = 400
        image_width  = 400

        number_of_color_channels = 3
        color_background         = (255, 255, 255)

        joint_0_arm_1 = (observation_space_vector[0],  observation_space_vector[1])
        joint_0_arm_2 = (observation_space_vector[2],  observation_space_vector[3])
        joint_1_arm_1 = (observation_space_vector[4],  observation_space_vector[5])
        joint_1_arm_2 = (observation_space_vector[6],  observation_space_vector[7])
        end_arm_1     = (observation_space_vector[8],  observation_space_vector[9])
        end_arm_2     = (observation_space_vector[10], observation_space_vector[11])
        cube          = (observation_space_vector[12], observation_space_vector[13])
        target        = (observation_space_vector[14], observation_space_vector[15])

        joint_0_arm_1 = self.normalization_conversion(joint_0_arm_1, image_height, image_width)
        joint_0_arm_2 = self.normalization_conversion(joint_0_arm_2, image_height, image_width)
        joint_1_arm_1 = self.normalization_conversion(joint_1_arm_1, image_height, image_width)
        joint_1_arm_2 = self.normalization_conversion(joint_1_arm_2, image_height, image_width)
        end_arm_1     = self.normalization_conversion(end_arm_1, image_height, image_width)
        end_arm_2     = self.normalization_conversion(end_arm_2, image_height, image_width)

        cube   = self.normalization_conversion(cube, image_height, image_width)
        target = self.normalization_conversion(target, image_height, image_width)

        size_rectangle      = 10
        half_size_rectangle = 5

        img = np.full((image_height, image_width, number_of_color_channels), color_background, dtype=np.uint8)

        cv2.line(img, (joint_0_arm_1[0], joint_0_arm_1[1]),
                      (joint_1_arm_1[0], joint_1_arm_1[1]),
                      (0, 0, 0), 2)

        cv2.line(img, (joint_0_arm_2[0], joint_0_arm_2[1]),
                      (joint_1_arm_2[0], joint_1_arm_2[1]),
                      (0, 0, 0), 2)

        cv2.line(img, (joint_1_arm_2[0], joint_1_arm_2[1]),
                      (end_arm_2[0], end_arm_2[1]),
                      (0, 0, 0), 2)

        cv2.line(img, (joint_1_arm_1[0], joint_1_arm_1[1]),
                      (end_arm_1[0], end_arm_1[1]),
                      (0, 0, 0), 2)

        cv2.rectangle(img,  (joint_0_arm_1[0] - half_size_rectangle, joint_0_arm_1[1] - half_size_rectangle),
                            (joint_0_arm_1[0] + half_size_rectangle, joint_0_arm_1[1] + half_size_rectangle),
                            (0, 255, 255), -1)

        cv2.rectangle(img,  (joint_0_arm_2[0] - half_size_rectangle, joint_0_arm_2[1] - half_size_rectangle),
                            (joint_0_arm_2[0] + half_size_rectangle, joint_0_arm_2[1] + half_size_rectangle),
                            (69, 139, 116), -1)

        cv2.rectangle(img,  (joint_1_arm_1[0] - half_size_rectangle, joint_1_arm_1[1] - half_size_rectangle),
                            (joint_1_arm_1[0] + half_size_rectangle, joint_1_arm_1[1] + half_size_rectangle),
                            (255, 228, 196), -1)

        cv2.rectangle(img,  (joint_1_arm_2[0] - half_size_rectangle, joint_1_arm_2[1] - half_size_rectangle),
                            (joint_1_arm_2[0] + half_size_rectangle, joint_1_arm_2[1] + half_size_rectangle),
                            (156, 102, 31), -1)

        cv2.rectangle(img, (end_arm_1[0] - half_size_rectangle, end_arm_1[1] - half_size_rectangle),
                           (end_arm_1[0] + half_size_rectangle, end_arm_1[1] + half_size_rectangle),
                           (255, 158, 160), -1)

        cv2.rectangle(img, (end_arm_2[0] - half_size_rectangle, end_arm_2[1] - half_size_rectangle),
                           (end_arm_2[0] + half_size_rectangle, end_arm_2[1] + half_size_rectangle),
                           (255, 100, 35), -1)

        cv2.rectangle(img, (target[0], target[1]), (target[0] + size_rectangle, target[1] + size_rectangle), (0, 0, 255), -1)
        cv2.rectangle(img, (cube[0], cube[1]), (cube[0] + size_rectangle, cube[1] + size_rectangle), (0, 255, 0), -1)
        return img


    def normalization_conversion(self, point, imag_h, imag_w):
        # convert values from camera frame to image to display's values
        # position_x = (point[0] - (-180)) * (imag_w - 0) / (180 - (-180))
        # position_y = (point[1] - (-120)) * (imag_h - 0) / (120 - (-120))
        position_x = (point[0] - (-300)) * (imag_w - 0) / (300 - (-300)) # 180
        position_y = (point[1] - (-350)) * (imag_h - 0) / (350 - (-350)) # 120
        return int(position_x), int(position_y)


