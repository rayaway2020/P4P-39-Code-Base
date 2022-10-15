#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: David Valencia
Date: 03/ 05 /2022
Description:
            Motor: MY_DXL ---> 'XL320'
            Motor features:
                -Bps = 1 Mbps
                -Min steps  = 0
                -Max steps  = 1023
                -Max Degree = 300
                -Resolution [deg/pulse] = 0.2930

            This script loads the parameters and internal configuration for each motor.
            To summarize:
                1) Set the address and values
                2) Open the USB port
                3) Enable Torque for each motor
                4) limit the Speed for each motor
                5) limit the Torque for each motor (very important)
                6) Initialize GroupSyncWrite instance
"""

import os
from dynamixel_sdk import *

if os.name == 'nt':
    import msvcrt
    def getch():
        return msvcrt.getch().decode()
else:
    import sys, tty, termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    def getch():
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


#  ------------------Initial Parameters---------------------------
# Address of each parameter. See the eManual for details about these values
ADDR_TORQUE_ENABLE    = 24
ADDR_GOAL_POSITION    = 30
ADDR_MOVING_SPEED     = 32
ADDR_TORQUE_LIMIT     = 35
ADDR_PRESENT_POSITION = 37

PROTOCOL_VERSION = 2.0  # Default for XL-320
BAUDRATE         = 1000000   # Default Baudrate of XL-320 is 1Mbps

# ID for each motor
DXL_ID_1 = 1
DXL_ID_2 = 2
DXL_ID_3 = 3
DXL_ID_4 = 4

# Use the actual port assigned to the U2D2.
DEVICENAME = '/dev/ttyUSB0'

# Configuration values
TORQUE_ENABLE          = 1     # Value for enabling the torque
TORQUE_DISABLE         = 0     # Value for disabling the torque
# DXL_MAX_VELOCITY_VALUE = 150   # Value for limited the speed. Max possible value=2047 meaning max speed
# DXL_MAX_TORQUE_VALUE   = 125   # It is the torque value of maximum output. 0 to 1,023 can be used

DXL_MAX_VELOCITY_VALUE = 150   # Value for limited the speed. Max possible value=2047 meaning max speed
DXL_MAX_TORQUE_VALUE   = 110   # It is the torque value of maximum output. 0 to 1,023 can be used

# Initialize PortHandler instance
# Set the port path
portHandler = PortHandler(DEVICENAME)

# Initialize PacketHandler instance
packetHandler = PacketHandler(PROTOCOL_VERSION)

# ------------------------Open port-------------------------
print("-------------------------------------------------------------")
if portHandler.openPort():
    print("Succeeded to open the port")
else:
    print("Failed to open the port")
    print("Press any key to terminate...")
    getch()
    quit()

print("-------------------------------------------------------------")
# -----------------------------------------------------------------


# ---------------Enable Torque Limit for each motor-----------------------------------------------------------
# This should be here because if torque is enable=1 we can not change the max torque again

dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID_1, ADDR_TORQUE_LIMIT, DXL_MAX_TORQUE_VALUE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully limited the TORQUE" % DXL_ID_1)

dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID_2, ADDR_TORQUE_LIMIT, DXL_MAX_TORQUE_VALUE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully limited the TORQUE" % DXL_ID_2)

dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID_3, ADDR_TORQUE_LIMIT, DXL_MAX_TORQUE_VALUE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully limited the TORQUE" % DXL_ID_3)

dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID_4, ADDR_TORQUE_LIMIT, DXL_MAX_TORQUE_VALUE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully limited the TORQUE" % DXL_ID_4)
    print("-------------------------------------------------------------")
# -----------------------------------------------------------------


# ---------------Enable Torque for each motor-----------------------
# Enable Dynamixel_1 Torque
dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID_1, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully enable torque" % DXL_ID_1)

# Enable Dynamixel_2 Torque
dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID_2, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully enable torque" % DXL_ID_2)

# Enable Dynamixel_3 Torque
dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID_3, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully enable torque" % DXL_ID_3)

# Enable Dynamixel_4 Torque
dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID_4, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully enable torque" % DXL_ID_4)
    print("-------------------------------------------------------------")
# -----------------------------------------------------------------


# ---------------Enable Moving Speed Limited for each motor---------------------------------------------------------

dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID_1, ADDR_MOVING_SPEED, DXL_MAX_VELOCITY_VALUE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully limited the speed" % DXL_ID_1)


dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID_2, ADDR_MOVING_SPEED, DXL_MAX_VELOCITY_VALUE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully limited the speed" % DXL_ID_2)


dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID_3, ADDR_MOVING_SPEED, DXL_MAX_VELOCITY_VALUE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully limited the speed" % DXL_ID_3)


dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID_4, ADDR_MOVING_SPEED, DXL_MAX_VELOCITY_VALUE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully limited the speed" % DXL_ID_4)
    print("-------------------------------------------------------------")


# ---------------------Initialize GroupSyncWrite instance --------------------
# Need this in order to move all the motor at the same time
# Initialize GroupSyncWrite instance ---> GroupSyncWrite(port, ph, start_address, data_length)

data_length    = 2  # data len of goal position and present position
groupSyncWrite = GroupSyncWrite(portHandler, packetHandler, ADDR_GOAL_POSITION, data_length)


