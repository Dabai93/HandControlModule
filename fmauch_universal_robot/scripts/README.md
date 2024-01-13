# Use gesture control UR5e robot

## setup roscore in terminal
roscore

## set up gazebo virtual environment and generate UR5e robot model
cd catkin/src/fmauch_universal_robot/ur_e_gazebo/launch/ur5e.launch
roslaunch UR5e.launch

## start handcontrol model 
cd catkin/src/fmauch_universal_robot/scripts
handcontrol_script.py



