class underXML:
    fingers=["5R","10R","15R","5F","10F","15F"]

    def __init__(self,fingertip,damping=5,kp=100000,f_slide=1, f_tor=.08, f_roll=0.01,solimp="0.95 0.99 0.0001",solref="0.001 1") -> None:
        if fingertip in self.fingers:
            tip=fingertip
        else:
            tip="10R"
        self.xml= f"""
        <!-- Copyright 2021 DeepMind Technologies Limited

     Licensed under the Apache License, Version 2.0 (the "License");
     you may not use this file except in compliance with the License.
     You may obtain a copy of the License at

         http://www.apache.org/licenses/LICENSE-2.0

     Unless required by applicable law or agreed to in writing, software
     distributed under the License is distributed on an "AS IS" BASIS,
     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
     See the License for the specific language governing permissions and
     limitations under the License.
   -->

   <mujoco model="underactuated">
    <!-- set some defaults for units and lighting -->
    <compiler angle="radian" meshdir="/home/david/test_bed_RL_robot/Test_bed_Code/Simulation/Under_STL_Meter"/>

    <option timestep="0.0002" iterations="50000" solver="Newton" cone="elliptic" tolerance="1e-10" gravity= "0 0  -9.80665"/>

    <size njmax="500" nconmax="100" nstack="2000"/>

    <visual>
      <rgba haze=".3 .3 .3 1"/>
    </visual>

    <default>
     
     <geom   solref='{solref}' solimp='{solimp}'/>
     <default class="finger">
      <geom contype='1' conaffinity='1' condim='6'  margin="1e-3" friction="{f_slide} {f_tor} {f_roll}"/>
    </default>
    
    <default class="Basejoint">
      <joint type="hinge" axis="0 0 1"  limited="true" range="-1.5708 1.5708"  stiffness="90" damping="{damping}" />
    </default>

    <default class="Distaljoin">
      <joint type="hinge" axis="0 0 1"  limited="true" range="-1.5708 -0.244346" springref="-0.244346" stiffness="160" damping="{damping}" />
    </default>

    <default class="sliderJoint">
      <joint type="slide" axis="0 1 0"  limited="true" range="-.0250 .0250" damping="100"  />
    </default>

    <default class="MainTendon">
      <tendon  stiffness="100000" damping="10"  solimplimit=".99 .99 0.0001" />
    </default>

  </default>


  <!-- import our stl files -->
  <asset>
    <mesh file="Sim_under_base.STL" />
    <mesh file="Sim_under_proximal.STL" />
    <mesh file="Sim_under_distal_{tip}.STL" />

    <texture type="skybox" builtin="gradient" rgb1=".4 .6 .8" rgb2="0 0 0" width="1" height="1"/>
    <texture name="groundplane" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 .2 .3" width="1" height="1"/>
    <material name="groundplane" texture="groundplane" texrepeat="1 1"/>  

  </asset>

 

  <worldbody>
   <!-- Ground-->
   <geom name="floor" contype="1" conaffinity="1" type="plane" pos="0 0 0" size=".25 .25 0.025" material="groundplane"/>

   <light directional="true" diffuse=".8 .8 .8" specular=".2 .2 .2" pos="0 0 5" dir="0 0 -1"/>

   <!-- start model -->

   <body name="cube" pos="0 .097 .04">
            <joint type="free"  damping="0.01"/>
            <geom name="testCube" type="box" size=".025 .025 .03" rgba=".5 .1 0.1 1"  mass=".086" contype='1' conaffinity='1' condim='6'  friction='1 0.005 0.001'/>
            <site name="Rightside" type="box" pos="0.025 0 0" size="0.001 0.025 0.028" rgba="0 .9 0 .1"/>
            <site name="Leftside" type="box" pos="-0.025 0 0" size="0.001 0.025 0.028" rgba="0 .9 0 .1"/>
            <site name="BottomRightside" type="box" pos="0.025 -.02 0" size="0.0006 0.006 0.028" rgba="1 .9 0 .1"/>
            <site name="BottomLeftside" type="box" pos="-0.025 -.02 0" size="0.0006 0.006 0.028" rgba="1 .9 0 .1"/>
            <geom name="BottomRightside" type="box" pos="0.025 -.02 0" size="0.0006 0.006 0.028" rgba="0 .9 0 .1"/>
            <geom name="BottomLeftside" type="box" pos="-0.025 -.02 0" size="0.0006 0.006 0.028" rgba="0 .9 0 .1"/>
        </body>

   <body name="base" pos="0 0 0">
    <geom name="base1" type="mesh" mesh="Sim_under_base" pos="0 0 0" contype="0" conaffinity="0"/>
    <inertial pos="0 0 0" mass="100" diaginertia="0 0 0"/>
    <geom name="baseLPiv" type="cylinder" fromto="-.0325 0 .030  -.0325 0 .050" size=".010" rgba="1 .9 .3 .4" contype="0" conaffinity="0"/>
    <geom name="baseRPiv" type="cylinder" fromto=".0325 0 .030  .0325 0 .050" size=".010" rgba="1 .9 .3 .4" contype="0" conaffinity="0"/>
    <site name="baseLI" pos="-.022 0 .0408" size=".0005" rgba="0 .7 0 1" />
    <site name="baseRI" pos=".022 0 .0408" size=".0005" rgba="1 .7 0 1" />
    <site name="baseLO" pos="-.044 0 .0408" size=".0005" rgba="0 .7 0 1" />
    <site name="baseRO" pos=".044 0 .0408" size=".0005" rgba="1 .7 0 1" />
    <site name="baseRPulley" pos=".021 -0.029 0.0408" size=".0005" rgba="1 .7 0 1" />
    <site name="baseLPulley" pos="-.021 -0.029 0.0408" size=".0005" rgba="0 .7 0 1" />


    <!-- Finger 1 model -->
    <body name="finger1" pos="-.0325 0 .0408" euler="0 0 0.174533">

      <geom name="prox1" type="mesh" mesh="Sim_under_proximal" pos="0 0 0"  />
      <inertial pos=".030 -.006 0" mass=".023" diaginertia="1 1 1"/>
      <joint name="fing1" class="Basejoint"/>
      <geom name="prox1Pin" type="cylinder" fromto="-.004 .030 .0165  -.004 .030 -.0165" size=".0015" rgba=".3 .9 .3 .4"/>

      <site name="halfProx1" pos=" .0014 .0171 0" size=".001" rgba="0 .7 0 1" />
      <site name="pin1" pos="-.006 .030 0" size=".001" rgba="0 .7 0 1" />
      <site name="pin1_1" pos="-.004 .030 0" size=".001" rgba="0 .7 0 1" />

      <body name="dist1" pos="0 .060 0">

        <geom name="distal1" type="mesh" mesh="Sim_under_distal_{tip}" pos="0 0 0" euler="0 0 0" class="finger"/>
        <inertial pos="0.0005 .0203 0" mass=".026" diaginertia="1 1 1"/>
        <joint name="dis1"  class="Distaljoin"/>

        <geom name="dis1Pin" type="cylinder" fromto="0 0 .010  0 0 -.010" size=".008" rgba=".3 .9 .3 .4" />
        <site name="dis1PinO" pos="-.007 -.007 0" size=".001" rgba="0 .7 0 1"/>
     
        <site name="dis1" pos=".008 .0015 0" size=".001" rgba="0 .7 0 1"/>
        <site name="dis1_1" pos="-.008 .008 0" size=".001" rgba="0 .7 0 1"/>
      </body>


    </body>
    <!-- Finger 2 model -->
    <body name="finger2" pos=".0325 0 .0408" euler="0 3.14159 0.174533">

      <geom name="prox2" type="mesh" mesh="Sim_under_proximal" pos="0 0 0"  />
      <inertial pos=".030 -.006 0" mass=".023" diaginertia="1 1 1"/>
      <joint name="fing2" class="Basejoint"/>
      <geom name="prox2Pin" type="cylinder" fromto="-.004 .030 .0165  -.004 .030 -.0165" size=".0015" rgba=".3 .9 .3 .4"/>

      <site name="halfProx2" pos=" .0014 .0171 0" size=".001" rgba="0 .7 0 1" />
      <site name="pin2" pos="-.006 .030 0" size=".001" rgba="0 .7 0 1" />
      <site name="pin2_1" pos="-.004 .030 0" size=".001" rgba="0 .7 0 1" />

      <body name="dist2" pos="0 .060 0">

        <geom name="distal2" type="mesh" mesh="Sim_under_distal_{tip}" pos="0 0 0" euler="0 0 0" class="finger"/>
        <inertial pos="0.0005 .0203 0" mass=".026" diaginertia="1 1 1"/>
        <joint name="dis2"  class="Distaljoin"/>

        <geom name="dis2Pin" type="cylinder" fromto="0 0 .010  0 0 -.010" size=".008" rgba=".3 .9 .3 .4" />
        <site name="dis2PinO" pos="-.007 -.007 0" size=".001" rgba="0 .7 0 1"/>
      
        <site name="dis2" pos=".008 .0015 0" size=".001" rgba="0 .7 0 1"/>
        <site name="dis2_1" pos="-.008 .008 0" size=".001" rgba="0 .7 0 1"/>
      </body>


    </body>

  </body>


</worldbody>



<tendon>
  <spatial name="MainLeft" width="0.0002" rgba=".95 .3 .3 1" class="MainTendon"  >
   <site site="baseLPulley"/>

   <geom geom="baseLPiv" sidesite="baseLI"/>
         <site site="halfProx1"/>
      <geom geom="prox1Pin" sidesite="pin1"/>

   <site site="dis1"/>
 </spatial>
 <spatial name="MainRight" width="0.0002" rgba=".95 .3 .3 1" class="MainTendon"  >
   <site site="baseRPulley"/>
      <geom geom="baseRPiv" sidesite="baseRI"/>
      <site site="halfProx2"/>
      <geom geom="prox2Pin" sidesite="pin2"/>
      <site site="dis2"/>
 </spatial>
</tendon>


<contact>

</contact>

<actuator>
  <position name="motorL" tendon="MainLeft"  ctrlrange="-100 160" kp="{kp}" ctrllimited="true"  />
  <position name="motorR" tendon="MainRight"  ctrlrange="-100 160" kp="{kp}" ctrllimited="true"  />
</actuator>

<sensor>
  <tendonpos name="tendonLeft" tendon="MainLeft"/>
  <tendonpos name="tendonRight" tendon="MainRight"/>

  <framepos objtype="body" objname="cube"/>
  <framequat objtype="body" objname="cube"/>

  <touch name="objectLeft" site="Leftside"/>
  <touch name="objectRight" site="Rightside"/>
  <touch name="LeftStop" site="BottomLeftside"/>
  <touch name="RightStop" site="BottomRightside"/>
  
  
</sensor>
</mujoco>   
        """
