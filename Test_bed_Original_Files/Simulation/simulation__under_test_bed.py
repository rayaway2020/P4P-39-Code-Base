'''
Description:
            Date ; 16/04/2022

            Modified simulation file. The original file "SimulationUnder" had certain settings
            that did not allow the simulation to be initialized. 

            I have briefly modified this code. I have changed:
            
            
            - sim=SimulationUnder(xml.xml,0) 0 --> 1 this should be 1 in order to start the simulation 
            - change the path in UnderXML.py --> around line 27
            - fix the save directory paths, specifically "UnderData" folder path 


'''
import os
from mujoco_py import load_model_from_xml, MjSim, MjViewer
import math
import os
import numpy as np
import pandas as pd
import glfw
from underXML import underXML
import matplotlib.pyplot as plt
import math


class GetOutOfLoop( Exception ):
    pass

#Convert quaternion from sensor to euler
# Code from https://automaticaddison.com/how-to-convert-a-quaternion-into-euler-angles-in-python/
def euler_from_quaternion(w, x, y, z):
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        x_rotation = math.atan2(t0, t1)
     
        t2 = 2.0 * (w * y - z * x)
        t2 = 1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        y_rotation = math.asin(t2)
     
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        z_rotation = math.atan2(t3, t4)
     
        return x_rotation, y_rotation, z_rotation # in radians


class SimulationUnder():
    defaultTendonLength=0.09611374

    delay=1000
    
    #set data name
    data_column_names=["motor1", "motor2" , "tendon1","tendon2", "x" ,"y", "zRotation",  "round"]

    #Intialise simulation and viewer
    def __init__(self,xml,view) -> None:
        #"/home/bci/Dropbox/1 Thesis/code/Simulation/Mujoco/model/AH-underactatedhand_New Sensor.xml"

        model = load_model_from_xml(xml)
        self.sim = MjSim(model)
        if view:
            self.viewer = MjViewer(self.sim)
            self.viewer._run_speed=64.0
        else:
            self.viewer=None
        self.showsim()

    #Close viewer window
    def close(self):
        if self.viewer is not None :
            glfw.destroy_window(self.viewer.window)

    #Simulate step
    def showsim(self):
        for i in range (self.delay):
            self.sim.step()
            if self.viewer is not None :
                self.viewer.render()

    #Send control commands to the simulatiion
    def ctrl_set_action(self, action):
        """For torque actuators it copies the action into mujoco ctrl field.
        For position actuators it sets the target relative to the current qpos.
        """
        if self.sim.data.ctrl is not None:
            for i in range(action.shape[0]):
                self.sim.data.ctrl[i] = action[i]
                
    def get_sensor_sensordata(self):
        #Data in array [Tendon1 , tendon 2 ,object X ,object Y, object Z ,quat (4 cells w,x,y,z), touchRight, TouchLeft , left stop, right stop]
        return self.sim.data.sensordata

    def get_position_sensordata(self):
        #Data in array [Tendon1 , tendon 2 ,object X ,object Y, object Z ,quat (4 cells w,x,y,z), touchRight, TouchLeft , left stop, right stop]
        return self.sim.data.sensordata[2:5] #[object X ,object Y, object Z]

    def get_rotation_sensordata(self):
        return self.sim.data.sensordata[5:9]  #[quat (4 cells w,x,y,z)]

    #This moves the actuators till the object is touched
    def moveToTouch(self):
        control=0
        data=self.get_sensor_sensordata()[9:11]
        
        while np.all(data <= 1000):
            
                control+=0.5
                self.ctrl_set_action(np.array([-control,-control]))
                self.showsim()
                data=self.get_sensor_sensordata()[-2:]
                
            
        return control

    #This moves the actuators till the object is in the start position
    def moveToStart(self):
        data=self.get_sensor_sensordata()[3]
        while data<0.088:
            self.sim.step()
            data=self.get_sensor_sensordata()[3]
        control=0
        count=0

        #Mimic reconfiguration of actual model
        while data>0.088:
            control+=0.005
            self.ctrl_set_action(np.array([-control,-control]))
            self.showsim()
            data=self.get_sensor_sensordata()[3]


        return control

    #This moves the object to the new intial position for each round
    def basetocontrol(self,basePosition,control):
        adjustement=0.01
        movement=control
        for i in range(basePosition):

            movement=control+i*adjustement
            motorvalues=np.array([-movement,-movement])
            self.ctrl_set_action(motorvalues)
            self.showsim()

        return movement


    #This moves the actuators till the object is touched
    def movesideways(self,control,direction,steps):
        data=pd.DataFrame(columns=self.data_column_names)
        #print(f'steps {steps} control {control} direction{direction}')
       
        controladjustment=0.001

        for i in range(steps):
            
            movement=direction*i*controladjustment
            motor1=(control+movement)
            motor2=(control-movement)
            #print(motor1 ,motor2)
            motorvalues=np.array([-motor1,-motor2])
            self.ctrl_set_action(motorvalues)
            
            self.showsim()
            
            sensorData=self.get_sensor_sensordata()
            _,_,zrotation=euler_from_quaternion(*sim.get_rotation_sensordata())

           #break if the object is lost
            #else record data
            if sensorData[4]>=0.031:
                print("break because of height")
                break
            elif sensorData[-2]>1 or sensorData[-1]>1:
                print("sensors hit")
                break

            data.loc[len(data)]=[motor1,motor2,sensorData[0],sensorData[1],sensorData[2],sensorData[3],zrotation,0]
           
        return data

    #Run a trajectory
    def runTragectory(self, base, steps, direction):
        done=0
        #reset the sim
        self.sim.reset()
        t = 0
        control=self.moveToStart()

        #move the object to the start position for each trial
        adjustment=self.basetocontrol(base,control)
        control= adjustment
        distance=self.get_sensor_sensordata()[3]

        #run until this distance - found by trial
        if distance <= 0.0725:
            done=1
            data=pd.DataFrame(columns=self.data_column_names)
            return data,done
            
        
        motorvalues=np.array([-control,-control])
        self.ctrl_set_action(motorvalues)
        self.showsim()

        data=self.movesideways(control,direction,steps)
        
        data["round"]=base
        return data , done

   
#Defines the arrays for testing multiple values
models=["5R","10R","15R"]
dam=[10,20,30,40,50,60,70,80,90,100]
f_slide=[1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]
f_tor=[0.5,0.05,0.005 ]
f_roll=[0.1,0.01,0.001,0.0001]
solref=[0.5]
sol=[0.95,0.96,0.97,0.98,0.99]

limp1=[[min,max] for min in sol for max in sol if min<=max]


#Cycle through all the arrays
for m in models:
    for damp in dam:
        for f in f_slide:
            for f_t in f_tor:
                for f_r in f_roll:
                    for ref in solref:
                        solr=f'0.001 {ref}' #make string for solref
                        for l in limp1:
                            
                            limp=f"{l[0]} {l[1]} 0.0001" #Make string for solimp

                            #Define the model and start simulator                    
                            xml=underXML(fingertip=m,damping=damp,f_slide=f,f_tor=f_t,f_roll=f_r ,solimp=limp, solref=solr)
                            sim=SimulationUnder(xml.xml,1) # this should be 1 in order to start the simulation
                            
                            round=f"{m}_{damp}_{f}_{f_t}_{f_r}_{limp}"
                            print(round)

                            #Creae the data frame to stor data
                            data=pd.DataFrame(columns=SimulationUnder.data_column_names)
                            steps=[20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 25, 25, 25, 25, 25, 27, 27, 27, 27, 27, 27, 27, 30, 30, 30, 30, 30, 32, 32, 32, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40]
                            directions=[1,-1]
                            done=0
                            index=0
                            try:
                                while not done:
                                    #run trajectories till end
                                    for d in directions:
                                        data2,done=sim.runTragectory(index,300,d)
                                        
                                        if done:
                                            break
                                        else:
                                            data=pd.concat([data,data2])
                                    index+=1
                                    
                                   #break if over 75 trials. something has gone wrong
                                    if index>5:
                                        break

                            #Close the sim and save and plot data
                            finally:
                                sim.close()

                                outdir =  "UnderData"   
                                if  not os.path.exists(outdir):
                                    os.mkdir(outdir)

                                data.to_csv(f"{outdir}/under_{m}_{damp}_{f}_{f_t}_{f_r}_{limp}.csv")
                                scatter=data.plot.scatter(x='x',y='y')
                                fig = scatter.get_figure()
                                fig.savefig(f"{outdir}/under_{m}_{damp}_{f}_{f_t}_{f_r}_{limp}.png")
                                plt.close(fig)