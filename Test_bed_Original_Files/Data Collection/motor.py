import dynamixel_sdk as dynamixel
from dynamixel_sdk import port_handler                     # Uses DYNAMIXEL SDK library



class Motor:
    # Protocol version
    PROTOCOL_VERSION            = 2                             # See which protocol version is used in the Dynamixel

    # Control Table
    ADDR_PRO_TORQUE_ENABLE      = 0                           # Control table address is different in Dynamixel model
    ADDR_PRO_GOAL_POSITION      = 0
    ADDR_PRO_PRESENT_POSITION   = 0
    ADDR_PRO_MOVING             = 0
    ADD_PRO_ERROR               = 0
    ADD_VELOCITY_LIMIT          = 0
    
    # Data Byte Length
    LEN_PRO_GOAL_POSITION       = 0
    LEN_PRO_PRESENT_POSITION    = 0

    # Communication data
    BAUDRATE                    = 57600
    # DEVICENAME                  = "COM3"                        # Check which port is being used on your controller
                                                                # ex) Windows: "COM1"   Linux: "/dev/ttyUSB0"
    
    TORQUE_ENABLE               = 1                             # Value for enabling the torque
    TORQUE_DISABLE              = 0                             # Value for disabling the torque

    COMM_SUCCESS                = 0                             # Communication Success result value
    COMM_TX_FAIL                = -1001                         # Communication Tx Failed

    dxl_comm_result = COMM_TX_FAIL                              # Communication result
    dxl_addparam_result = 0                                     # AddParam result
    dxl_getdata_result = 0                                      # GetParam result

    velocity_limit = 25

    port_handler    = None
    packet_handler  = None
    groupwrite_num  = None
    groupread_num   = None

    dxl_error = 0                                               # Dynamixel error

    motors=[]

    #Initialise the motors
    def initMotorControl(self):
        comm_result=0
        if self.port_handler.openPort():
            print("Succeeded to open the port!")
        else:
            comm_result=1
            print("Failed to open the port!")

        # Set port baudrate
        if self.port_handler.setBaudRate(self.BAUDRATE):
            print("Succeeded to change the baudrate!")
        else:
            comm_result=2
            print("Failed to change the baudrate!")

        return comm_result
    
    #Check the connection with the motors
    def checkConnection(self):

        for m in self.motors:
            _, result, error=self.packet_handler.ping(self.port_handler,m)
            self.packet_handler.getTxRxResult(result)
            if result != self.COMM_SUCCESS:
                print("Com error")
                raise Exception("Com error") 
            self.packet_handler.getRxPacketError(error)
            if error != 0:
                print("Packet error")
                raise Exception("Packet error") 
            print("connection good")

    #Check for hardware errors
    def checkHardwareError(self):
        errorList=[]
        for m in self.motors:
            HardwareError, comm_result, error = self.packet_handler.read1ByteTxRx(self.port_handler, m, self.ADD_PRO_ERROR)
            if comm_result != self.COMM_SUCCESS:
                print("Com error")
            if error != 0:
                print("Packet error")
            errorList.append(HardwareError)
        return errorList

    #Connect and setup grouprwite
    def connect(self):
        # Initialize PortHandler Structs
        # Set the port path
        # Get methods and members of PortHandlerLinux or PortHandlerWindows
        self.port_handler = dynamixel.PortHandler(self.DEVICENAME)

        # Initialize PacketHandler Structs
        self.packet_handler=dynamixel.PacketHandler(self.PROTOCOL_VERSION)

        # Initialize Groupsyncwrite instance
        self.groupwrite_num = dynamixel.GroupSyncWrite(self.port_handler, self.packet_handler, self.ADDR_PRO_GOAL_POSITION, self.LEN_PRO_GOAL_POSITION)

    #enable torque control
    def enableTorque(self):
        for m in self.motors:
            comm_result, error= self.packet_handler.write1ByteTxRx(self.port_handler, m, self.ADDR_PRO_TORQUE_ENABLE, self.TORQUE_ENABLE)
            self.errorCheck(comm_result,error, m,"Enable torque")
        
    #Disable Torque control
    def disableTorque(self):
        for m in self.motors:
            comm_result, error = self.packet_handler.write1ByteTxRx(self.port_handler, m, self.ADDR_PRO_TORQUE_ENABLE, self.TORQUE_DISABLE)
            self.errorCheck(comm_result,error, m, "Disable torque")

    
    #Check if the motors are moving
    def isMoving(self):
        moving=0
        for m in self.motors:
            is_Moving, comm_result, error = self.packet_handler.read1ByteTxRx(self.port_handler, m, self.ADDR_PRO_MOVING)
            self.errorCheck(comm_result,error, m, "Is moving")
            moving+=is_Moving
        return moving

    #Write position to the motor
    def writePosition(self,motorID,goal_position):
        if self.LEN_PRO_GOAL_POSITION == 2:
           comm_result, error = self.packet_handler.write2ByteTxRx(self.port_handler, motorID, self.ADDR_PRO_GOAL_POSITION, goal_position)
        elif self.LEN_PRO_GOAL_POSITION ==4:
            comm_result, error = self.packet_handler.write4ByteTxRx(self.port_handler, motorID, self.ADDR_PRO_GOAL_POSITION, goal_position)
        else:
            print("goal position length is not correct")
            return 1

        self.errorCheck(comm_result,error, motorID, "write position")
        return comm_result

    #Write positon to a group of motors
    def writeGroupPosition(self,goal_position_array):
        for index, pos in enumerate(goal_position_array):
            # Add Dynamixel#1 goal position value to the Syncwrite storage

            dxl_addparam_result=self.groupwrite_num.addParam( self.motors[index],  pos)
            if dxl_addparam_result != 1:
                print("[ID:%03d] groupSyncWrite addparam failed" % (self.motors[index]))
        # Syncwrite goal position
        self.groupwrite_num.txPacket()
        # Clear syncwrite parameter storage
        self.groupwrite_num.clearParam()

    #Get the position of a motor
    def trackPosition(self, motorID):
        if self.LEN_PRO_PRESENT_POSITION == 2:
            present_position, comm_result, error = self.packet_handler.read2ByteTxRx(self.port_handler, motorID, self.ADDR_PRO_PRESENT_POSITION)
        elif self.LEN_PRO_PRESENT_POSITION == 4:
            present_position, comm_result, error = self.packet_handler.read4ByteTxRx(self.port_handler, motorID, self.ADDR_PRO_PRESENT_POSITION)
        else:
            print("present position length is not correct")
            return 1

        self.errorCheck(comm_result,error, motorID, "Track Position")
        return present_position

    #Get all positions
    def getAllPositions(self):
        allPosition=[]
        for m in self.motors:
            allPosition.append(self.trackPosition(m))
        return allPosition

    #write the velocity limit of the motor
    def writeVelocityLimit(self):
        print(f'setting velocity limit {self.velocity_limit}')
        for m in self.motors:
            if self.LEN_PRO_GOAL_POSITION == 2:
                comm_result, error = self.packet_handler.write2ByteTxRx(self.port_handler, m, self.ADD_VELOCITY_LIMIT, self.velocity_limit)
            elif self.LEN_PRO_GOAL_POSITION ==4:
                comm_result, error = self.packet_handler.write4ByteTxRx(self.port_handler, m, self.ADD_VELOCITY_LIMIT, self.velocity_limit)
            else:
                print("Failed to set velocity limit")
                return 1

            self.errorCheck(comm_result,error ,m, "write Veolicty")

    #Stub for turning on the LED
    def turnOnLED(self,color):
        pass

    #Shut down and clear the port
    def shutdown(self):
        self.disableTorque()
        self.turnOnLED(0)
        self.port_handler.clearPort()

    #Check error codes
    def errorCheck(self,comm_result,error,motorID,location):
        return comm_result
        if comm_result != self.COMM_SUCCESS:
            print(str(location)+": %d %s" % (motorID,self.packet_handler.getTxRxResult(comm_result)))
        elif error != 0:
            print(str(location)+": %d %s" % (motorID, self.packet_handler.getRxPacketError(error)))
        return comm_result

    


#XL 320
class XL320(Motor):
    dxl_home_position=512

    def __init__(self,port) -> None:
        self.DEVICENAME=port
        # Control table address
        self.ADDR_PRO_TORQUE_ENABLE      = 24                           # Control table address is different in each Dynamixel model refer to documenation
        self.ADDR_PRO_GOAL_POSITION      = 30
        self.ADDR_PRO_PRESENT_POSITION   = 37
        self.ADDR_PRO_LED                = 25
        self.ADDR_PRO_MOVING             = 49
        self.ADD_PRO_ERROR               = 50
        self.ADD_VELOCITY_LIMIT          = 32
        self.ADDR_PRO_P                  = 29
        self.ADDR_PRO_I                  = 28
        self.ADDR_PRO_D                  = 27
        self.ADDR_PRO_Goal_Velocity      = 32
        
        # Data Byte Length
        self.LEN_PRO_GOAL_POSITION       = 2
        self.LEN_PRO_PRESENT_POSITION    = 2

        self.BAUDRATE                    = 57600

        self.velocity_limit = 2

        # Default setting
        DXL1_ID                     = 1                             # Dynamixel ID: 1
        DXL2_ID                     = 2                             # Dynamixel ID: 2
        DXL3_ID                     = 3                             # Dynamixel ID: 2
        DXL4_ID                     = 4                             # Dynamixel ID: 2
        self.motors=[DXL1_ID,DXL2_ID,DXL3_ID,DXL4_ID]

        
        self.DXL_MINIMUM_POSITION_VALUE  = 0                       # Dynamixel will rotate between this value
        self.DXL_MAXIMUM_POSITION_VALUE  = 1023                        # and this value (note that the Dynamixel would not move when the position value is out of movable range. Check e-manual about the range of the Dynamixel you use.)

        self.dxl5_lower_position= 500
        self.dxl5_raise_position= 790
        self.DXL_MOVING_STATUS_THRESHOLD = 1                            # Dynamixel moving status threshold

        self.dxl_home_position=512
        # self.dxl1_home_position = 525                                  # Present position
        # self.dxl2_home_position = 552
        # self.dxl3_home_position = 500
        # self.dxl4_home_position = 472
        
        self.dxl1_present_position = 0                                   # Present position
        self.dxl2_present_position = 0
        self.dxl3_present_position = 0
        self.dxl4_present_position = 0
        # Initialize PortHandler Structs
        # Set the port path
        # Get methods and members of PortHandlerLinux or PortHandlerWindows
        self.connect()

    #enable torque
    def enableTorque(self):
        comm_result, error= self.packet_handler.write1ByteTxRx(self.port_handler, 5, self.ADDR_PRO_TORQUE_ENABLE, self.TORQUE_ENABLE)
        self.errorCheck(comm_result,error, 5, "enable Torque")
        return super().enableTorque()

    #disable torque
    def disableTorque(self):
        comm_result, error= self.packet_handler.write1ByteTxRx(self.port_handler, 5, self.ADDR_PRO_TORQUE_ENABLE, self.TORQUE_DISABLE)
        self.errorCheck(comm_result,error, 5 , "disable Torque")
        return super().disableTorque()
   
   #set the pid controls
    def setPID(self,P,I,D):
        for m in self.motors:
            comm_result, error = self.packet_handler.write1ByteTxRx(self.port_handler, m, self.ADDR_PRO_P,P)
            self.errorCheck(comm_result,error, m, "set P")
            
            comm_result, error = self.packet_handler.write1ByteTxRx(self.port_handler, m, self.ADDR_PRO_I,I)
            self.errorCheck(comm_result,error, m, "set I")
            
            comm_result, error = self.packet_handler.write1ByteTxRx(self.port_handler, m, self.ADDR_PRO_D,D)
            self.errorCheck(comm_result,error, m, "set D")

    # Set goal velocity
    def setGoalVelocity(self,value):
         for m in self.motors:
            comm_result, error = self.packet_handler.write2ByteTxRx(self.port_handler, m, self.ADDR_PRO_Goal_Velocity,value)
            self.errorCheck(comm_result,error, m, "set velocity goal")

    #Turn on all LEDS
    def turnOnLED(self,color):
        for m in self.motors:
            comm_result, error = self.packet_handler.write1ByteTxRx(self.port_handler, m, self.ADDR_PRO_LED, color)
            self.errorCheck(comm_result,error, m, "turn On LED")
        comm_result, error = self.packet_handler.write1ByteTxRx(self.port_handler, 5, self.ADDR_PRO_LED, color)
        self.errorCheck(comm_result,error, 5,"turn On LED")
    #check for erros
    def checkHardwareError(self):
        errorList=super().checkHardwareError()
        for val in errorList:
            if val & 1:
                print("Overeload error")
            if val & 2:
                print("overheating error")
            if val & 4:
                print("input voltage error")
        return 
    
    #Raise the object
    def raiseObject(self):
        comm_result, error = self.packet_handler.write2ByteTxRx(self.port_handler, 5, self.ADDR_PRO_GOAL_POSITION, self.dxl5_raise_position)
        self.errorCheck(comm_result,error, 5,"raise Object")
        return comm_result

    #Lower the object
    def lowerObject(self):
        comm_result, error = self.packet_handler.write2ByteTxRx(self.port_handler, 5, self.ADDR_PRO_GOAL_POSITION, self.dxl5_lower_position)
        self.errorCheck(comm_result,error, 5,"Lower Object")
        return comm_result

    # Open the gripper to release the object
    def openGripper(self):
        base=25
        distal=57
        positions=[XL320.dxl_home_position +base,XL320.dxl_home_position +distal,XL320.dxl_home_position -base,XL320.dxl_home_position -distal]
        self.writeGroupPosition(positions)



class XM430(Motor):
    dxl1_home_position = 2865            #1875                     # Present position
    dxl2_home_position = 2300             #2285
    dxl3_raise_position = 3650
    dxl3_lower_position= 2900

    def __init__(self,port) -> None:
    # XM430 W350

        # Control table address
        self.ADDR_PRO_TORQUE_ENABLE      = 64                           # Control table address is different in Dynamixel model
        self.ADDR_PRO_GOAL_POSITION      = 116
        self.ADDR_PRO_PRESENT_POSITION   = 132
        self.ADDR_PRO_MOVING             = 122
        self.ADD_PRO_ERROR               = 70
        self.ADD_PRO_LED                 = 65
        self.ADD_VELOCITY_LIMIT          = 44

        # Data Byte Length
        self.LEN_PRO_GOAL_POSITION       = 4
        self.LEN_PRO_PRESENT_POSITION    = 4

        self.BAUDRATE                    = 57600
        self.DEVICENAME                  = port

        # Default setting
        DXL1_ID                     = 1                             # Dynamixel ID: 1
        DXL2_ID                     = 2                             # Dynamixel ID: 2
        self.motors=[DXL1_ID,DXL2_ID]

        self.velocity_limit = 25

        self.dxl1_present_position = 0                                   # Present position
        self.dxl2_present_position = 0

        self.DXL_MINIMUM_POSITION_VALUE  = 0                      # Dynamixel will rotate between this value
        self.DXL_MAXIMUM_POSITION_VALUE  = 4095                      # and this value (note that the Dynamixel would not move when the position value is out of movable range. Check e-manual about the range of the Dynamixel you use.)
        self.DXL_MOVING_STATUS_THRESHOLD = 1                          # Dynamixel moving status threshold

        self.connect()

    #enable torque
    def enableTorque(self):
        comm_result, error= self.packet_handler.write1ByteTxRx(self.port_handler, 3, self.ADDR_PRO_TORQUE_ENABLE, self.TORQUE_ENABLE)
        self.errorCheck(comm_result,error, 3, "enable Torque")
        return super().enableTorque()

    #Disable Torque
    def disableTorque(self):
        comm_result, error= self.packet_handler.write1ByteTxRx(self.port_handler, 3, self.ADDR_PRO_TORQUE_ENABLE, self.TORQUE_DISABLE)
        self.errorCheck(comm_result,error, 3, "disable Torque")
        return super().disableTorque()

    #Check for hardware errors
    def checkHardwareError(self):
        errorList=super().checkHardwareError()
        for val in errorList:
            if val & 1:
                print("Overeload error")
            if val & 4:
                print("Overheating error")
            if val & 8:
                print("encoder error")
            if val & 16:
                print("Electrical shock error")
            if val & 32:
                print("Overload error")
        return 

    #Turn on LED
    def turnOnLED(self,ON):
        for m in self.motors:
            comm_result, error = self.packet_handler.write1ByteTxRx(self.port_handler, m, self.ADD_PRO_LED , ON)
            self.errorCheck(comm_result,error, m,"turn On LED")
        comm_result, error = self.packet_handler.write1ByteTxRx(self.port_handler, 3, self.ADD_PRO_LED , ON)
        self.errorCheck(comm_result,error, 3,"turn On LED")
    
    #Raise the object
    def raiseObject(self):
        comm_result, error = self.packet_handler.write4ByteTxRx(self.port_handler, 3, self.ADDR_PRO_GOAL_POSITION, self.dxl3_raise_position)
        self.errorCheck(comm_result,error, 3, "raise Object")
        return comm_result

    #lower the object
    def lowerObject(self):
        comm_result, error = self.packet_handler.write4ByteTxRx(self.port_handler, 3, self.ADDR_PRO_GOAL_POSITION, self.dxl3_lower_position)
        self.errorCheck(comm_result,error, 3,"Lower Object")
        return comm_result

    #Open the gripper
    def openGripper(self):
        openvalue=110
        positions=[self.dxl1_home_position+openvalue, self.dxl2_home_position-openvalue]
        self.writeGroupPosition(positions)