#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Description:

"""

import os

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


from dynamixel_sdk import *  # Uses Dynamixel SDK library


ADDR_TORQUE_ENABLE      = 24
ADDR_GOAL_POSITION      = 30
ADDR_MOVING_SPEED       = 32
ADDR_TORQUE_LIMIT       = 35
ADDR_PRESENT_POSITION   = 37


# Protocol and Bps
PROTOCOL_VERSION = 2.0
BAUDRATE         = 1000000  # Default Baudrate of XL-320 is 1Mbps

DXL_ID_1 = 1
DXL_ID_2 = 2
DXL_ID_3 = 3
DXL_ID_4 = 4

# Use the actual port assigned to the U2D2.
DEVICENAME = '/dev/ttyUSB0'

TORQUE_ENABLE  = 1  # Value for enabling the torque
TORQUE_DISABLE = 0  # Value for disabling the torque

# Speed values
DXL_MAX_VELOCITY_VALUE = 160   # Max possible value=2047
DXL_LIM_TORQUE_VALUE   = 180    # Max possible value=1023


# GOAL VALUES FOR EACH MOTOR
index = 0
id_1_dxl_goal_position = [300, 700]  # Goal position for motor 1
id_2_dxl_goal_position = [300, 700]  # Goal position for motor 2

id_3_dxl_goal_position = [300, 700]  # Goal position for motor 3
id_4_dxl_goal_position = [300, 700]  # Goal position for motor 2

portHandler   = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)

# ------------------------Open port-------------------------
if portHandler.openPort():
    print("Succeeded to open the port")
else:
    print("Failed to open the port")
    print("Press any key to terminate...")
    getch()
    quit()
# ----------------------------------------------------------


# ---------------Enable Torque Limit for each motor-----------------------------------------------------------
# This should be here because if torque is enable=1 we can not change the max torque again

dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID_1, ADDR_TORQUE_LIMIT, DXL_LIM_TORQUE_VALUE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully limited the TORQUE" % DXL_ID_1)

dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID_2, ADDR_TORQUE_LIMIT, DXL_LIM_TORQUE_VALUE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully limited the TORQUE" % DXL_ID_2)

dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID_3, ADDR_TORQUE_LIMIT, DXL_LIM_TORQUE_VALUE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully limited the TORQUE" % DXL_ID_3)

dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID_4, ADDR_TORQUE_LIMIT, DXL_LIM_TORQUE_VALUE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully limited the TORQUE" % DXL_ID_4)

# ---------------Enable Torque for each motor-----------------------------------------------------------

# Enable Dynamixel#1 Torque
dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID_1, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)

if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully connected" % DXL_ID_1)

# Enable Dynamixel#2 Torque
dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID_2, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully connected" % DXL_ID_2)

# Enable Dynamixel#3 Torque
dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID_3, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully connected" % DXL_ID_3)

# Enable Dynamixel#4 Torque
dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID_4, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully connected" % DXL_ID_4)

# ---------------Enable Moving Speed for each motor-----------------------------------------------------------

dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID_1, ADDR_MOVING_SPEED,
                                                          DXL_MAX_VELOCITY_VALUE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully limited the speed" % DXL_ID_1)

dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID_2, ADDR_MOVING_SPEED,
                                                          DXL_MAX_VELOCITY_VALUE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully limited the speed" % DXL_ID_2)

dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID_3, ADDR_MOVING_SPEED,
                                                          DXL_MAX_VELOCITY_VALUE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully limited the speed" % DXL_ID_3)

dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID_4, ADDR_MOVING_SPEED,
                                                          DXL_MAX_VELOCITY_VALUE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully limited the speed" % DXL_ID_4)

# ------------------------------------------------------------------------------------------------------------


# ---------------------Initialize GroupSyncWrite instance --------------------

# Initialize GroupSyncWrite instance ---> GroupSyncWrite(port, ph, start_address, data_length)

data_length = 2  # data length goal position  and present position

groupSyncWrite = GroupSyncWrite(portHandler, packetHandler, ADDR_GOAL_POSITION, data_length)

# syncwrite test start

while 1:

    print(index)
    print("Press any key to continue! (or press ESC to quit!)")
    if getch() == chr(0x1b):
        break

    # The Size of the goal position is 2 bytes

    param_goal_position_1 = [DXL_LOBYTE(id_1_dxl_goal_position[index]), DXL_HIBYTE(id_1_dxl_goal_position[index])]
    param_goal_position_2 = [DXL_LOBYTE(id_2_dxl_goal_position[index]), DXL_HIBYTE(id_2_dxl_goal_position[index])]
    param_goal_position_3 = [DXL_LOBYTE(id_3_dxl_goal_position[index]), DXL_HIBYTE(id_3_dxl_goal_position[index])]
    param_goal_position_4 = [DXL_LOBYTE(id_4_dxl_goal_position[index]), DXL_HIBYTE(id_4_dxl_goal_position[index])]

    # --- Add the goal position value to the Syn parameter, motor ID1 ----
    dxl_addparam_result = groupSyncWrite.addParam(DXL_ID_1, param_goal_position_1)

    if dxl_addparam_result != True:
        print("[ID:%03d] groupSyncWrite addparam failed" % DXL_ID_1)
        quit()

    # --- Add the goal position value to the Syn parameter, motor ID2 ----
    dxl_addparam_result = groupSyncWrite.addParam(DXL_ID_2, param_goal_position_2)
    if dxl_addparam_result != True:
        print("[ID:%03d] groupSyncWrite addparam failed" % DXL_ID_2)
        quit()

    dxl_addparam_result = groupSyncWrite.addParam(DXL_ID_3, param_goal_position_3)
    if dxl_addparam_result != True:
        print("[ID:%03d] groupSyncWrite addparam failed" % DXL_ID_3)
        quit()

    dxl_addparam_result = groupSyncWrite.addParam(DXL_ID_4, param_goal_position_4)
    if dxl_addparam_result != True:
        print("[ID:%03d] groupSyncWrite addparam failed" % DXL_ID_4)
        quit()


    # ---- Transmits packet (goal position) to the motors
    dxl_comm_result = groupSyncWrite.txPacket()
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))

    # Clear syncwrite parameter storage
    groupSyncWrite.clearParam()

    # Change goal position index
    if index == 0:
        index = 1
    else:
        index = 0


# Disable communication and close the port

# ---------------Disable Torque for each motor-----------------------
# Enable Dynamixel_1 Torque
dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID_1, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully disable torque" % DXL_ID_1)

# Enable Dynamixel_2 Torque
dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID_2, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully disable torque" % DXL_ID_2)

# Enable Dynamixel_3 Torque
dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID_3, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully disable torque" % DXL_ID_3)

# Enable Dynamixel_4 Torque
dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID_4, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully disable torque" % DXL_ID_4)

# Close port
portHandler.closePort()
print("Succeeded to close the USB port ")