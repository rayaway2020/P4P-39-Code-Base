# from PIL import Image, ImageDraw, ImageOps
#
# from main_rl_env_translation_v2 import RL_ENV
# import numpy as np
# import cv2
#
# def plot_state_space(d, observation_space):
#     image_height = 300
#     image_width = 300
#     size = 10
#     cube_targ_size = 10
#
#     number_of_color_channels = 3
#     color = (202, 202, 202)
#
#     img = np.full((image_height, image_width, number_of_color_channels), color, dtype=np.uint8)
#     print(observation_space)
#     joint_0_arm_1 = (int(observation_space[0][0]), int(observation_space[0][1]))
#     joint_0_arm_2 = (int(observation_space[0][2]), int(observation_space[0][3]))
#     joint_1_arm_1 = (int(observation_space[0][4]), int(observation_space[0][5]))
#     joint_1_arm_2 = (int(observation_space[0][6]), int(observation_space[0][7]))
#     end_arm_1 = (int(observation_space[0][8]), int(observation_space[0][9]))
#     end_arm_2 = (int(observation_space[0][10]), int(observation_space[0][11]))
#     cube = (int(observation_space[0][12]), int(observation_space[0][13]))
#     target = (int(observation_space[0][14]), int(observation_space[0][15]))
#
#     cv2.rectangle(img, (cube[0], cube[1]), (cube[0] + cube_targ_size, cube[1] + cube_targ_size), (0, 255, 0), -1)
#     cv2.rectangle(img, (target[0], target[1]), (target[0] + cube_targ_size, target[1] + cube_targ_size),
#                   (0, 0, 255), -1)
#
#     cv2.line(img, (joint_1_arm_1[0] + 5, joint_1_arm_1[1] + 5), (end_arm_1[0] + 5, end_arm_1[1] + 5), (0, 0, 0), 2)
#     cv2.line(img, (joint_1_arm_2[0] + 5, joint_1_arm_2[1] + 5), (end_arm_2[0] + 5, end_arm_2[1] + 5), (0, 0, 0), 2)
#
#     cv2.rectangle(img, (joint_0_arm_1[0], joint_0_arm_1[1]), (joint_0_arm_1[0] + size, joint_0_arm_1[1] + size),
#                   (255, 0, 128), -1)
#     cv2.rectangle(img, (joint_1_arm_1[0], joint_1_arm_1[1]), (joint_1_arm_1[0] + size, joint_1_arm_1[1] + size),
#                   (0, 0, 153), -1)
#     cv2.rectangle(img, (end_arm_1[0], end_arm_1[1]), (end_arm_1[0] + size, end_arm_1[1] + size), (204, 102, 0), -1)
#
#     cv2.rectangle(img, (joint_0_arm_2[0], joint_0_arm_2[1]), (joint_0_arm_2[0] + size, joint_0_arm_2[1] + size),
#                   (128, 255, 128), -1)
#     cv2.rectangle(img, (joint_1_arm_2[0], joint_1_arm_2[1]), (joint_1_arm_2[0] + size, joint_1_arm_2[1] + size),
#                   (128, 128, 255), -1)
#     cv2.rectangle(img, (end_arm_2[0], end_arm_2[1]), (end_arm_2[0] + size, end_arm_2[1] + size), (128, 0, 128), -1)
#
#     return img
#
# width  = 200
# height = 200
# img  = Image.new(mode="RGB", size=(width, height), color=(202, 202, 202))
# draw = ImageDraw.Draw(img)
#
# import time
# import psutil
#
# for _ in range(0, 10):
#     # env    = RL_ENV()
#     # state  = env.state_space_function()
#     # img1 = plot_state_space(draw, state)
#     # img.show()
#     # draw = ImageDraw.Draw(img1)
#     image_height = 300
#     image_width = 300
#     size = 10
#     cube_targ_size = 10
#
#     number_of_color_channels = 3
#     color = (202, 202, 202)
#     img1 = np.full((image_height, image_width, number_of_color_channels), color, dtype=np.uint8)
#
#     cv2.imshow("input", img1)
#     # cv2.imshow("State visual", crop_image)
#     # cv2.waitKey(10)
#     time.sleep(2.0)
#     # env.reset_env()
#     # for proc in psutil.process_iter():
#     #     # print(proc)
#     #     if proc.name().__contains__("video"):
#     #         proc.kill()



# from PIL import Image, ImageDraw, ImageOps
#
# from main_rl_env_translation_v2 import RL_ENV
#
#
# def plot_state_space(d, states):
#     size = 10
#     print(states[0][8], states[0][9])
#     d.rectangle([(states[0][8], states[0][9]), (states[0][8] + size, states[0][9] + size)], fill=(0, 255, 0, 0))
#     d.rectangle([(states[0][10], states[0][11]), (states[0][10] + size, states[0][11] + size)], fill=(255, 0, 0, 0))
#
#
# width  = 200
# height = 200
# img  = Image.new(mode="RGB", size=(width, height), color=(202, 202, 202))
# draw = ImageDraw.Draw(img)
#
# import time
# import psutil
#
# for _ in range(0, 10):
#     env    = RL_ENV()
#     state  = env.state_space_function()
#     plot_state_space(draw, state)
#     img.show()
#     time.sleep(2.0)
#
#     for proc in psutil.process_iter():
#         if proc.name() == "Image Viewer":
#             proc.kill()
