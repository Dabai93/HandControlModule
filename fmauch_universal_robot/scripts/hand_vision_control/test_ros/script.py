import numpy as np
from math import pi
from ur_ikfast import ur_kinematics

import rospy
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal, JointTrajectoryControllerState, GripperCommandAction, GripperCommandGoal
from trajectory_msgs.msg import JointTrajectoryPoint



class UR5e:
    def __init__(self,robotNr,controllerID,BaseShift,BaseRotation):

        #Inverse Kinematics
        self.kin = {}
        self.kin['theta_min'] = np.array([-2*pi, -2.6,       0,   -4.0, -1.6, -2*pi])  
        self.kin['theta_max'] = np.array([ 2*pi,     0,  2.6,  -1.0,  1.6,  2*pi])

        #ROS Action Client
        self.ROS = {}
        if controllerID == 1:
            self.ROS['controller'] = '/scaled_vel_joint_traj_controller/follow_joint_trajectory'
            self.ROS['subscriber'] = '/scaled_vel_joint_traj_controller/state'
        elif controllerID == 2:
            self.ROS['controller'] = '/vel_joint_traj_controller/follow_joint_trajectory'
            self.ROS['subscriber'] = '/vel_joint_traj_controller/state'
        else:
            print('Choose one of the available controllers!')
        self.ROS['jointNames'] = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        if (robotNr == 1) & (controllerID == 1):
            self.ROS['robotNr'] = '/robot030'
        elif (robotNr == 2) & (controllerID == 1):
            self.ROS['robotNr'] = '/robot029'
        else:
            self.ROS['robotNr'] = '/robot' + str(robotNr)
        self.ROS['gripper'] = '/gripper_controller/follow_joint_trajectory'
        self.ROS['gripperNames'] = ['robotiq_85_left_knuckle_joint']
        self.ROS['gripperStroke'] = [0, 0.8]
        self.ROS['points'] = 10

        # Gripper
        self.Length = {}     
        self.Length['lGripper'] = 0.177
        self.Length['lGripperOpen'] = 0.162
        self.Length['lGripperFinger'] = 0.037


        self.Base = {}
        self.Base['Shift'] = BaseShift
        self.Base['Rotation'] = BaseRotation
        self.ur5e_arm = ur_kinematics.URKinematics('ur5e')
        self.intialization_robot_control()


    def inverse_kinematics(self,Pos):
        Pos = np.array(Pos).reshape((-1,1))
        #print(Pos)
        if Pos.shape[0] == 4:
            xyz = Pos[0:3] - self.Base['Shift']
        else:
            xyz = Pos - self.Base['Shift']
        xyz = xyz.reshape(-1,1)

        # Dimension of the gripper
        dim = {}
        dim['gripper'] = np.array([0,0,-self.Length['lGripperOpen']]).reshape((-1,1))

        # Rotation matrix of the endeffector
        Rot = {}
        Rot['EF'] = np.array([[-0.00650204, -0.99997807, -0.0012669],
                        [-0.99996871,  0.00649626,  0.0045156],
                        [-0.00450727,  0.00129622, -0.99998897]])
        Rot['gripper'] = np.array([[1,0,0],[0,1,0],[0,0,1]])
        H = {}
        H['gripper'] = np.array([[1,0,0,0],
                              [0,1,0,0],
                              [0,0,1,-self.Length['lGripperOpen']],
                              [0,0,0,1]])
        d = np.array([[0,0,0,1]])
        # TCP Position
        H['all'] = np.concatenate([np.concatenate([Rot['EF'],xyz],axis = 1),d],axis = 0)

        # Homogenous tranformation matrix for the gripper
        if Pos.shape[0] == 4:
            theta = -Pos[3,0]
            rotz = np.array([[np.cos(theta), -np.sin(theta), 0, 0],
                            [np.sin(theta),np.cos(theta),0,0],
                            [0,0,1,0],
                            [0,0,0,1]])
            H['all'] = np.matmul(H['all'],rotz) 
        
        # Homogenous tranformation matrix for the ee
        H['EF'] = np.matmul(np.linalg.inv(H['gripper']),H['all'])

        qs = self.ur5e_arm.inverse(H['EF'][0:3,:],True)

        q = [x for x in qs if (x<= self.kin['theta_max']).all() & (x >= self.kin['theta_min']).all()]
        q = q[0]
        
        EF60 = self.ur5e_arm.forward(q,'matrix')
        EF60 = np.concatenate([EF60,d],axis = 0)
        EF_tcp0 = np.matmul(H['gripper'],EF60)

        if np.linalg.norm(EF_tcp0[0:3,3].reshape((-1,1))-xyz) > 1e-2:
            print('Wrong solution of the inverse kinematics')
        else:
            q_desired = q + np.array([self.Base['Rotation']-np.pi,0,0,0,0,0])
        return np.array(q_desired).reshape((1,-1))

    def forward_kinematics(self,state):
        mat = self.ur5e_arm.forward(state,'matrix')
        return mat 
    

    def intialization_robot_control(self):

        self.actClient = actionlib.SimpleActionClient(self.ROS['robotNr']+self.ROS['controller'], FollowJointTrajectoryAction)
        self.actClient.feedback_cb = None
        self.actClient.wait_for_server()
        # Create subscriber
        self.mySub = rospy.Subscriber(self.ROS['robotNr']+self.ROS['subscriber'],JointTrajectoryControllerState)
        # Create message
        self.msg = FollowJointTrajectoryGoal()
        # Create publisher
        self.msg.trajectory.joint_names = self.ROS['jointNames']
        self.msg.trajectory.points.append(JointTrajectoryPoint())
        # Create publisher with flexible number of messages
        self.msgNum = FollowJointTrajectoryGoal()
        self.msgNum.trajectory.joint_names = self.ROS['jointNames']
        self.Npoints = self.ROS['points']
        self.trackingPoints = []
        for i in range(0,self.Npoints):
            self.trackingPoints.append(JointTrajectoryPoint())
        # Set initial gripper state to be fully open
        self.gazebo = True
        self.gripper_gazbeo_init()

    def transitionGazebo(self,t_span,trajectory):
        targetState = trajectory
        # Package ROS message and send to the robot
        self.msg.trajectory.points[0].time_from_start = rospy.Duration(t_span[1]-t_span[0])
        self.msg.trajectory.points[0].positions = targetState[0:6]
        self.msg.trajectory.points[0].velocities = targetState[6:12]
        #Publish message
        self.actClient.send_goal(self.msg)
    

    def gripper_gazbeo_init(self):
        self.pub_gripper = actionlib.SimpleActionClient(self.ROS['robotNr']+'/gripper_controller_gripperaction/gripper_cmd', GripperCommandAction)
        self.pub_gripper.wait_for_server()
        rospy.sleep(0.2)
        self.gripper_gazebo_control(self.ROS['gripperStroke'][1])
        rospy.sleep(0.5)
        self.gripper_gazebo_control(self.ROS['gripperStroke'][0])  
        rospy.sleep(3)

    def gripper_gazebo_control(self,pos):
        if self.gazebo:
            goal = GripperCommandGoal()
            goal.command.position = pos   # From 0.0 to 0.8
            goal.command.max_effort = -1.0  # Do not limit the effort
            self.pub_gripper.send_goal(goal)
            self.pub_gripper.wait_for_result()
            return self.pub_gripper.get_result()

    def openGripperGazebo(self):
        self.gripper_gazebo_control(self.ROS['gripperStroke'][0]) 
        rospy.sleep(2.4)
        # pass

    def closeGripperGazebo(self):
        self.gripper_gazebo_control(self.ROS['gripperStroke'][1]-0.57) 
        rospy.sleep(2.4)
        # pass

    
    



        

