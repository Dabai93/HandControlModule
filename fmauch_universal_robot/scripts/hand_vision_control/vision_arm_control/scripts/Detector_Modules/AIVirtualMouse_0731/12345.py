import cv2

cap = cv2.VideoCapture(1) # 0通常是默认摄像头的ID

while(True):
    ret, frame = cap.read() # 读取一帧图像
    cv2.imshow('frame', frame) # 显示图像
    if cv2.waitKey(1) & 0xFF == ord('q'): # 按q键退出
        break

cap.release()
cv2.destroyAllWindows()
