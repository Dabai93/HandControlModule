import cv2
import mediapipe as mp  # refer to "google.github.io/mediapipe/" for details
import time
import math

class handDetector():
    def __init__(self, mode=False, maxHands=2, modelC=1, detectionCon=0.8, trackCon=0.8):
        self.mode = mode
        self.maxHands = maxHands
        self.modelC = modelC
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands # https://google.github.io/mediapipe/solutions/hands.html
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelC, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20] # 提示Id, 5个元素分别代表大拇指到小拇指的节点号

    # 在输入的img中检测手部，将手部各关节节点标出并返回img
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        #print(self.results.multi_handedness)  # 获取检测结果中的左右手标签并打印
        #print(self.results)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    # 得到21个手部节点在屏幕中的坐标
    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    # print(id, cx, cy)
                    self.lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 8, (0, 0, 255), cv2.FILLED)
        return self.lmList # len(lmList)=21, len(lmList[i])=3

    # 检测每个手指是否伸出
    def fingersUp(self):
        fingers = []
        # 大拇指
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1] + 10:
            fingers.append(1) # 大拇指指尖节点的x坐标 > 大拇指第二个节点的x坐标(+10) 则认为大拇指伸出
        else:
            fingers.append(0)

        # 其余手指
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1) # 其余四个手指指尖节点的y坐标 < 对应手指第三个节点的y坐标 则认为对应手指伸出
            else:
                fingers.append(0)

        # totalFingers = fingers.count(1)
        return fingers # 5个元素对应从大拇指到小拇指 e.g. fingers=[0,1,0,0,0]代表只有食指伸出

    # 计算img中p1和p2节点之间的距离，返回距离length
    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
            length = math.hypot(x2 - x1, y2 - y1) # p1和p2间的欧几里得距离

        return length, img, [x1, y1, x2, y2, cx, cy]

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(1)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)        # 检测手势并画上骨架信息

        lmList = detector.findPosition(img)  # 获取得到坐标点的列表
        if len(lmList) != 0:
            print(lmList[4])

        h, w, c = img.shape
        draw = True
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
                              (0, 255, 0), 2)



        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, 'fps:' + str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow('Image', img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()