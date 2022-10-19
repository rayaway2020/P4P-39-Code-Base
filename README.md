# test_bed_RL_robot
Files and code of  Test beds for Learning Robotic Manipulation Strategies

Originals file and some modification by my own

## Notes on Hardware Setup

If you want to set up a new gripper, make sure you check the following:

- Install Dynamixel Wizard 2.0. Instructions can be found here https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_wizard2/ 
- When connecting motors, make sure all the cables are working and connected properly. 
- When setting up cameras, make sure you do calibration. You can do this under `camera_calibration_files` folder
- Use Dynamixel to change the ID of gripper. (In "ID")
- Use Dynamixel to test the angle. (In "Goal Position", make sure 0 degree returns the gripper to exactly the middle)
- Under Dynamixel ("Hardware Error Status"), make sure you only leave Bit1 Overheating Error Checked. 


## Install Software Packages

Make sure you are using Linux (Preferably Ubuntu-LTS 20.04) and installed Python3 and pip3

Under project root folder, run `pip3 install -r requirements.txt`

You also need to install Dynamixel SDK for Python. Detailed Instructions can be found here 
https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_sdk/library_setup/python_linux/#python-linux

## Check USB Ports

Run this command to check the USB status for motor.
`ls /dev/ttyUSB*`

Run this command to check the camera status.
`ls /dev/ttyvideo*`

Before running the file, make sure you checked above parameters in any Python File you want to run. 

To make sure, specifically, check these files:
- File you want run e.g. `DQN_Real_Robot.py`
- `vision_utilities.py`
- `main_rl_env_translation`
- `motor_utilities.py`

Sometimes you may need to give permission to these ports. For example:
`sudo chmod 777 /dev/ttyUSB0`

## How To run

Go to the Project Folder (We worked on robot_translation_v2_full_aruco)

`cd robot_translation_v2_full_aruco`

Run any file you want. The filename is self-explanatory
`python3 {filename}`


