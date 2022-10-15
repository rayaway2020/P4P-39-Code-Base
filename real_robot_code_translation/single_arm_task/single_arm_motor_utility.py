import time

from single_arm_motor_initializ import *  # --> set up motor configuration (speed, torque, etc...1)


def move_motor_step(id_1_dxl_goal_position, id_2_dxl_goal_position):

    # This function move the motor i.e. take the actions
    param_goal_position_1 = [DXL_LOBYTE(id_1_dxl_goal_position), DXL_HIBYTE(id_1_dxl_goal_position)]
    param_goal_position_2 = [DXL_LOBYTE(id_2_dxl_goal_position), DXL_HIBYTE(id_2_dxl_goal_position)]

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

        if ((abs(id_1_dxl_goal_position - present_step_pos_serv_1) < 8) and (
                abs(id_2_dxl_goal_position - present_step_pos_serv_2) < 8)):
            print("Action Completed")
            break

        end_time = time.time()
        timer = end_time - start_time

        if timer >= 2.0:
            print("time over, next action")
            break




def read_servo_position(motor_id):
    dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read2ByteTxRx(portHandler, motor_id, ADDR_PRESENT_POSITION)
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

    # Values in degrees
    tetha_1_arm_1 = pos_m1_arm_1 * 0.29326
    tetha_2_arm_1 = pos_m2_arm_1 * 0.29326


    # IMPORTANT, need these values in order to match the equations
    tetha_1_arm_1 = tetha_1_arm_1 - 60
    tetha_2_arm_1 = 150 - tetha_2_arm_1

    return tetha_1_arm_1, tetha_2_arm_1