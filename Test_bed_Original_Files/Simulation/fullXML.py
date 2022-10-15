class fullXML:
  fingers=["5R","10R","15R","5F","10F","15F"]

  def __init__(self,fingertip,damping=80,kp=1000,f_slide=1, f_tor=0.005, f_roll=0.0001,solimp="0.95 0.99 0.0001",solref="0.001 1") -> None:
      if fingertip in self.fingers:
          tip=fingertip
      else:
          tip="10R"
      self.xml= f"""<!-- Copyright 2021 DeepMind Technologies Limited

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
    <compiler angle="radian" meshdir="/home/bci/Dropbox/1 Thesis/code/Simulation/Mujoco/model/Full_STL_Meter"/>

    <option timestep="0.0002" iterations="50000" solver="Newton" cone="elliptic" tolerance="1e-10" gravity= "0 0  -9.80665"/>

    <size njmax="500" nconmax="100" nstack="2000"/>

    <visual>
      <rgba haze=".3 .3 .3 1"/>
    </visual>

    <default>
     
     <geom   solref="{solref}" solimp="{solimp}"/>
     <joint type="hinge" axis="0 0 1" damping="{damping}"/>
     <position  kp="{kp}" ctrllimited="true"/>

     <default class="finger">
      <geom contype='1' conaffinity='1' condim='6'  margin="1e-3"  friction="{f_slide} {f_tor} {f_roll}"/>
    </default>
    

  </default>


  <!-- import our stl files -->
  <asset>
    <mesh file="Sim_full_base.STL" />
    <mesh file="Sim_full_proximal.STL" />
    <mesh file="Sim_full_distal_{tip}.STL" />

     <texture type="skybox" builtin="gradient" rgb1=".4 .6 .8" rgb2="0 0 0" width="1" height="1"/>
    <texture name="groundplane" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 .2 .3" width="1" height="1"/>
    <material name="groundplane" texture="groundplane" texrepeat="1 1"/>  

  </asset>

 

  <worldbody>
   <!-- Ground-->
   <geom name="floor" contype="1" conaffinity="1" type="plane" pos="0 0 0" size=".25 .25 0.025" material="groundplane"/>

   <light directional="true" diffuse=".8 .8 .8" specular=".2 .2 .2" pos="0 0 5" dir="0 0 -1"/>

    <!-- start model -->

   <body name="cube" pos="0 .97 .04">
            <joint type="free"  damping="0.01"/>
            <geom name="testCube" type="box" size=".025 .025 .03" rgba=".5 .1 0.1 1"  mass=".086" contype='1' conaffinity='1' condim='6'  friction='1 0.005 0.001'/>
            <site name="Rightside" type="box" pos="0.025 0 0" size="0.001 0.025 0.028" rgba="0 .9 0 .1"/>
            <site name="Leftside" type="box" pos="-0.025 0 0" size="0.001 0.025 0.028" rgba="0 .9 0 .1"/>
            <site name="BottomRightside" type="box" pos="0.025 -.02 0" size="0.0006 0.006 0.028" rgba="1 .9 0 .1"/>
            <site name="BottomLeftside" type="box" pos="-0.025 -.02 0" size="0.0006 0.006 0.028" rgba="1 .9 0 .1"/>

        </body>

   <body name="base" pos="0 0 0">
    <geom name="base1" type="mesh" mesh="Sim_full_base" pos="0 0 0" contype="0" conaffinity="0"/>
    <inertial pos="0 0 0" mass="100" diaginertia="0 0 0"/>


    <!-- Finger 1 model -->
    <body name="finger1" pos="-.0335 0 .050" euler="0 0 0">

         <geom name="prox1" type="mesh" mesh="Sim_full_proximal" pos="0 0 0" euler="0 0 1.5708" />
          <inertial pos="0 0 0" mass=".065" diaginertia="1 1 1"/>

        <joint name="fing1" limited="true" range="-1.5708 1.5708"/>
     
      
        <body name="dist1" pos="0 .060 0" euler="0 3.14159 0">

        <geom name="distal1" type="mesh" mesh="Sim_full_distal_{tip}" pos="0 0 0" euler="0 3.14159 0" class="finger"/>
        <inertial pos="0.00075 .0273 0" mass=".022" diaginertia="1 1 1"/>
        <joint name="dis1"  limited="true" range="-1.5708 1.5708"/>

 
         </body>

    </body>

    <!-- Finger 2 model -->
    <body name="finger2" pos=".0335 0 .050" euler="0 0 0">

        <geom name="prox2" type="mesh" mesh="Sim_full_proximal" pos="0 0 0" euler="0 0 1.5708"   />
        <inertial pos="0 0 0" mass=".065" diaginertia="1 1 1"/>
        <joint name="fing2" limited="true" range="-1.5708 1.5708"/>
     
        <body name="dist2" pos="0 .060 0" euler="0 3.14159 0">
            <geom name="distal2" type="mesh" mesh="Sim_full_distal_{tip}" pos="0 0 0"  class="finger"/>
            <inertial pos="0.00075 .0273 0" mass=".022" diaginertia="1 1 1"/>
            <joint name="dis2"   limited="true" range="-1.5708 1.5708"/>
        </body>
      
    </body>



  </body>


</worldbody>

<actuator>
  <position name="Base1"    joint="fing1"   ctrlrange='-2 2'  />
  <position name="Distal1"  joint="dis1"    ctrlrange="-1.5708 1.5708"       />
  <position name="Base2"    joint="fing2"   ctrlrange='-2 2'  />
  <position name="Distal2"  joint="dis2"    ctrlrange=" -1.5708 1.5708"       />

</actuator>


<sensor>
    <jointpos  name="base1" joint="fing1" />
    <jointpos  name="distal1" joint="dis1" />
    <jointpos  name="base2" joint="fing2" />
    <jointpos  name="distal2" joint="dis2" />

    <framepos objtype="body" objname="cube"/>
    <framequat   objtype="body" objname="cube"/>

    <touch name="objectLeft" site="Leftside"/>
    <touch name="objectRight" site="Rightside"/>
    <touch name="LeftStop" site="BottomLeftside"/>
    <touch name="RightStop" site="BottomRightside"/>
    
</sensor>


</mujoco>   """