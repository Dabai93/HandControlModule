import cv2
import HandTrackingModule as htm
import autopy # autopy的介绍 https://blog.csdn.net/sandalphon4869/article/details/90272247
import numpy as np
import time
import pyautogui
#################################
wCam, hCam = 640, 480 # 设置opencv窗口的尺寸
frameR = 100
smoothening = 3
#################################
cap = cv2.VideoCapture(0)  # 若使用笔记本自带摄像头则编号为0  若使用外接摄像头 则更改为1或其他编号
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

detector = htm.handDetector()
wScr, hScr = autopy.screen.size() # 获取电脑屏幕的尺寸，用于食指在cv窗口坐标和电脑屏幕坐标之间的转换
print(wScr, hScr)

while True:
    success, img = cap.read()
    # 1. 检测手部 得到手指关键点坐标
    img = detector.findHands(img) # 手的各个关节节点(共21个节点)在img中被标出
    #cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - int(frameR*1.8)), (0, 255, 0), 2,  cv2.FONT_HERSHEY_PLAIN) # 画绿色的框
    lmList = detector.findPosition(img, draw=False) # len(lmList)=21, len(lmList[i])=3, e.g. lmList[8]=[8,xx,yy]代表食指指尖(8号)在cv窗口中的坐标为(xx,yy)
    #print(lmList)
    h, w, c = img.shape
    draw=True
    myHand = {}
    ## lmList
    mylmList = []
    xList = []
    yList = []
    if len(lmList) != 0:
        for id, lm in enumerate(lmList):
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
            cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                          (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                          (255, 0, 255), 2)
    # 2. 判断食指和中指是否伸出
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:] # 食指指尖在opencv窗口中的坐标
        x2, y2 = lmList[12][1:] # 中指指尖在opencv窗口中的坐标
        fingers = detector.fingersUp() # 判断手指是否伸出，5个元素对应从大拇指到小拇指 e.g. fingers=[0,1,0,0,0]代表只有食指伸出
        # print(fingers)

        # 3. 若只有食指伸出 则进入移动模式
        if fingers[1] and fingers[0] == False and fingers[2] == False and fingers[3] == False and fingers[4] == False:
            # 4. 坐标转换： 将食指在窗口坐标转换(等比缩放)为鼠标在桌面的坐标
            # 得到鼠标坐标(mouse_x,mouse_y) [线性插值函数 val = np.interp(x,arr1,arr2); arr1为横坐标值，arr2为纵坐标值，根据arr1和arr2将x映射到val]
            mouse_x = np.interp(x1, (frameR, wCam - frameR), (0, wScr))#将上方画的绿色框的宽度映射到屏幕宽度，将窗口横坐标x1映射为屏幕横坐标mouse_x
            mouse_y = np.interp(y1, (frameR, hCam - int(frameR*1.8)), (0, hScr))#绿色框的高度->屏幕高度，窗口纵坐标y1->屏幕纵坐标mouse_y

            # smoothening values
            clocX = plocX + (mouse_x - plocX) / smoothening
            clocY = plocY + (mouse_y - plocY) / smoothening
            print(wScr - clocX, clocY)
            autopy.mouse.move(np.max([wScr - clocX - 1e-6, 1e-6]), np.max([clocY, 1e-6])) # wScr - clocX 为了在横向方向上和手指移动方向镜像; 1e-6的目的是防止位置超出边界ValueError: Point out of bounds

            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED) # 在食指指尖部位画圆
            cv2.putText(img, "move", (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                        2, (255, 0, 255), 2)
            plocX, plocY = clocX, clocY

        # 5. 若食指和中指都伸出 则检测指头距离 距离够短则对应鼠标点击
        if fingers[1] and fingers[2] and fingers[0] == False and fingers[3] == False and fingers[4] == False:
            length, img, pointInfo = detector.findDistance(8, 12, img) # 8节点是食指指尖、12节点是中指指尖
            if length < 40:
                cv2.circle(img, (pointInfo[4], pointInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                pyautogui.leftClick()
                cv2.putText(img, "click left", (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                            2, (255, 0, 255), 2)

                # autopy.mouse.click()
                # autopy.mouse.toggle(None, True)
                # autopy.mouse.click(at.mouse.Button.LEFT, 3)
            # else:
                # autopy.mouse.toggle(None, False)
                # pyautogui.scroll(-100)

        # 6. 若食指和大拇指都伸出 则鼠标滚轮向下翻滚
        if fingers[0] and fingers[1] and fingers[2] == False and fingers[3] == False and fingers[4] == False:
            cv2.circle(img, (lmList[4][1], lmList[4][2]), 15, (0, 255, 0), cv2.FILLED) # 大拇指尖绘制圆
            cv2.circle(img, (lmList[8][1], lmList[8][2]), 15, (255, 0, 255), cv2.FILLED) # 食指指尖绘制圆
            pyautogui.scroll(-100) # 正值向上，负值向下
            cv2.putText(img, "scroll down", (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                        2, (255, 0, 255), 2)

        # 7. 若食指和小拇指都伸出 则鼠标滚轮向上翻滚
        if fingers[1] and fingers[4] and fingers[0] == False and fingers[2] == False and fingers[3] == False:
            cv2.circle(img, (lmList[20][1], lmList[20][2]), 15, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, (lmList[8][1], lmList[8][2]), 15, (255, 0, 255), cv2.FILLED)
            pyautogui.scroll(100)  # 正值向上，负值向下
            cv2.putText(img, "scroll up", (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                        2, (255, 0, 255), 2)

        # 8. 若四指伸出大拇指不伸 则点击鼠标右键
        if fingers[0] == False and fingers[1] and fingers[2] and fingers[3] and fingers[4]:
            cv2.circle(img, (lmList[8][1], lmList[8][2]), 15, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, (lmList[12][1], lmList[12][2]), 15, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, (lmList[16][1], lmList[16][2]), 15, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, (lmList[20][1], lmList[20][2]), 15, (0, 255, 0), cv2.FILLED)
            pyautogui.rightClick()
            #pyautogui.press('enter')
            cv2.putText(img, "click right", (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                        2, (255, 0, 255), 2)
        # 9. 若五个手指都伸出或者握拳，则手势识别为“no gesture”
        if fingers[0] and fingers[1] and fingers[2] and fingers[3] and fingers[4]:
            cv2.putText(img, "no gesture", (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                        2, (255, 0, 255), 2)

        if fingers[0]== False and fingers[1]== False and fingers[2]== False and fingers[3]== False and fingers[4]== False:
            cv2.putText(img, "no gesture", (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                        2, (255, 0, 255), 2)


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'fps:{int(fps)}', [15, 25], cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)