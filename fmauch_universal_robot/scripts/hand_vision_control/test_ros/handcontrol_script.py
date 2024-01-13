import rospy
rospy.init_node('handcontrol_script')
import cv2
import time
import math
from script import UR5e
import numpy as np
import tf.transformations as tr
from colorama import Fore, Back, Style, init
import sys
sys.path.append('/home/lzn/catkin/src/hand_vision_control/vision_arm_control/scripts') # this path should be changed 
from Detector_Modules.HandDetectorModule import HandDetector as hdm

Ts = 0.5
controllerID = 2 
robotNr = 1
BaseShift =  np.array([0.0, 0.0, 0.0]).reshape(-1,1)
BaseRotation = 0
gripper = False
Setpoints = {}
Np = 15
init(autoreset=True)

robot  = UR5e(robotNr,controllerID,BaseShift,BaseRotation)


def main(fps_cap=60, show_fps=True, source=0):


    assert fps_cap >= 1, f"fps_cap should be at least 1\n"
    assert source >= 0, f"source needs to be greater or equal than 0\n"

    ctime = 0  # current time (used to compute FPS)
    ptime = 0  # past time (used to compute FPS)
    prev_time = 0  # previous time variable, used to set the FPS limit

    fps_lim = fps_cap  # FPS upper limit value, needed for estimating the time for each frame and increasing performances

    time_lim = 1. / fps_lim  # time window for each frame taken by the webcam

    #initialize hand and pose detector objects
    HandDet = hdm()
    #PoseDet = pdm()
    cv2.setUseOptimized(True)  # enable OpenCV optimization

    # capture the input from the default system camera (camera number 0)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():  # if the camera can't be opened exit the program
        print("Cannot open camera")
        exit()
    while True:  # infinite loop for webcam video capture

        # computed  delta time for FPS capping
        delta_time = time.perf_counter() - prev_time

        ret, frame = cap.read()  # read a frame from the webcam

        if not ret:  # if a frame can't be read, exit the program
            print("Can't receive frame from camera/stream end")
            break

        if delta_time >= time_lim:  # if the time passed is bigger or equal than the frame time, process the frame
            prev_time = time.perf_counter()

            # compute the actual frame rate per second (FPS) of the webcam video capture stream, and show it
            ctime = time.perf_counter()
            fps = 1.0 / float(ctime - ptime)
            ptime = ctime

            frame = HandDet.findHands(frame=frame, draw=True)
            #frame = PoseDet.findPose(frame=frame, draw=False)
            #pose_lmlist = PoseDet.findPosePosition(frame=frame, draw=False)
            hand_lmlist, frame = HandDet.findHandPosition(
            frame=frame, hand_num=0, draw=False)
            h, w, c = frame.shape
            ###
            ###      
            draw=True
            myHand = {}
            ## lmList
            mylmList = []
            xList = []
            yList = []
            if len(hand_lmlist) != 0:
                for id, lm in enumerate(hand_lmlist):
                    px, py = int(lm[1]), int(lm[2])
                    mylmList.append([px, py])
                    xList.append(px)
                    yList.append(py)
                ## bbox
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH
                cx, cy = bbox[0] + (bbox[2] // 2), \
                    bbox[1] + (bbox[3] // 2)

                myHand["lmList"] = mylmList
                myHand["bbox"] = bbox
                myHand["center"] = (cx, cy)
                ## draw
                if draw:
                    cv2.rectangle(frame, (bbox[0] - 20, bbox[1] - 20),
                                (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                                (0, 255, 0), 2)
                
                fingers = HandDet.fingersUp()
                if fingers[0] and fingers[1] and fingers[2] and fingers[3] and fingers[4]:
                    cv2.putText(frame, "raise", (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                            2, (255, 0, 255), 2)

                if fingers[0]== False and fingers[1]== False and fingers[2]== False and fingers[3]== False and fingers[4]== False:
                    cv2.putText(frame, "fall", (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                            2, (255, 0, 255), 2)
                
            if len(hand_lmlist) > 0:
                frame, aperture = HandDet.findHandAperture(
                        frame=frame, verbose=True, show_aperture=True)
                print("the value of aperture is" + " " + str(aperture))
                joint_velocity = np.array([.0, .0, .0, .0, .0, .0])
                pos = [0.3,0.2,aperture/200] # [x=0.3,y=0.2,z=0.4]
                joint_angles = robot.inverse_kinematics(pos)
                joint_dynamic = np.concatenate([joint_angles[0],joint_velocity],axis = 0)



                robot.transitionGazebo([0.0, 4*Ts],joint_dynamic)
                rospy.sleep(10)


            if show_fps:
                cv2.putText(frame, "FPS:" + str(round(fps, 0)), (10, 400), cv2.FONT_HERSHEY_PLAIN, 2,
                            (255, 255, 255), 1)

            # show the frame on screen
            cv2.imshow("Frame (press 'q' to exit)", frame)

        # if the key "q" is pressed on the keyboard, the program is terminated
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


    return


if __name__ == '__main__':
    main(fps_cap=30, show_fps=True, source=0)










