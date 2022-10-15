

from motor_initialization import *


def read_servo_position(motor_id):

    dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read2ByteTxRx(portHandler, motor_id,
                                                                                   ADDR_PRESENT_POSITION)
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl_error))
    return dxl_present_position


def get_angles():
    # Arm 1
    # position in steps values
    pos_m1_arm_1 = read_servo_position(DXL_ID_1)
    pos_m2_arm_1 = read_servo_position(DXL_ID_2)

    # Arm 2
    # position in steps values
    pos_m3_arm_2 = read_servo_position(DXL_ID_3)
    pos_m4_arm_2 = read_servo_position(DXL_ID_4)

    # Values in degrees
    tetha_1_arm_1 = pos_m1_arm_1 * 0.29326
    tetha_2_arm_1 = pos_m2_arm_1 * 0.29326

    tetha_1_arm_2 = pos_m3_arm_2 * 0.29326
    tetha_2_arm_2 = pos_m4_arm_2 * 0.29326

    # IMPORTANT, need these values in order to match the equations
    tetha_1_arm_1 = tetha_1_arm_1 - 60
    tetha_1_arm_2 = tetha_1_arm_2 - 60

    tetha_2_arm_1 = 150 - tetha_2_arm_1
    tetha_2_arm_2 = 150 - tetha_2_arm_2

    return tetha_1_arm_1, tetha_2_arm_1, tetha_1_arm_2, tetha_2_arm_2


def move_motor_step(id_1_dxl_goal_position, id_2_dxl_goal_position, id_3_dxl_goal_position, id_4_dxl_goal_position):
    # This function move the motor i.e. take the actions

    param_goal_position_1 = [DXL_LOBYTE(id_1_dxl_goal_position), DXL_HIBYTE(id_1_dxl_goal_position)]
    param_goal_position_2 = [DXL_LOBYTE(id_2_dxl_goal_position), DXL_HIBYTE(id_2_dxl_goal_position)]
    param_goal_position_3 = [DXL_LOBYTE(id_3_dxl_goal_position), DXL_HIBYTE(id_3_dxl_goal_position)]
    param_goal_position_4 = [DXL_LOBYTE(id_4_dxl_goal_position), DXL_HIBYTE(id_4_dxl_goal_position)]

    # --- Add the goal position value to the GroupSync, motor ID1 ----
    dxl_addparam_result = groupSyncWrite.addParam(DXL_ID_1, param_goal_position_1)
    if dxl_addparam_result != True:
        print("[ID:%03d] groupSyncWrite addparam failed" % DXL_ID_1)
        quit()

    # --- Add the goal position value to the GroupSync, motor ID2 ----
    dxl_addparam_result = groupSyncWrite.addParam(DXL_ID_2, param_goal_position_2)
    if dxl_addparam_result != True:
        print("[ID:%03d] groupSyncWrite addparam failed" % DXL_ID_2)
        quit()

    # --- Add the goal position value to the GroupSync, motor ID3 ----
    dxl_addparam_result = groupSyncWrite.addParam(DXL_ID_3, param_goal_position_3)
    if dxl_addparam_result != True:
        print("[ID:%03d] groupSyncWrite addparam failed" % DXL_ID_3)
        quit()

    # --- Add the goal position value to the GroupSync, motor ID4 ----
    dxl_addparam_result = groupSyncWrite.addParam(DXL_ID_4, param_goal_position_4)
    if dxl_addparam_result != True:
        print("[ID:%03d] groupSyncWrite addparam failed" % DXL_ID_4)
        quit()

    # ---- Transmits packet (goal positions) to the motors
    dxl_comm_result = groupSyncWrite.txPacket()
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
    # Clear syncwrite parameter storage
    groupSyncWrite.clearParam()

    start_time = time.time()
    timer = 0

    while True:

        present_step_pos_serv_1 = read_servo_position(DXL_ID_1)
        present_step_pos_serv_2 = read_servo_position(DXL_ID_2)
        present_step_pos_serv_3 = read_servo_position(DXL_ID_3)
        present_step_pos_serv_4 = read_servo_position(DXL_ID_4)

        if ((abs(id_1_dxl_goal_position - present_step_pos_serv_1) < 8) and
                (abs(id_2_dxl_goal_position - present_step_pos_serv_2) < 8) and
                (abs(id_3_dxl_goal_position - present_step_pos_serv_3) < 8) and
                (abs(id_4_dxl_goal_position - present_step_pos_serv_4) < 8)):
            print("Action Completed")
            break

        end_time = time.time()
        timer = end_time - start_time

        if timer >= 1.5:
            print("time over, next action")
            break


def motor_terminate():
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
