import cv2
import mediapipe as mp
import time

cap=cv2.VideoCapture(0)

#新建手部检测对象
mpHands = mp.solutions.hands
hands=mpHands.Hands()

#新建绘点对象
mpDraw=mp.solutions.drawing_utils

#用来计算帧率
pTime=0
cTime=0

while True:
    success,img=cap.read()

    #将BGR转为RGB
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    #传入手部图片进行检测
    results=hands.process(imgRGB)

    # print(results.multi_hand_landmarks)

    #当有检测结果时
    if results.multi_hand_landmarks:
        #每组检测结果可能识别出多个手
        for handLms in results.multi_hand_landmarks:
            #获取检测结果点的xy坐标
            for id , lm in enumerate(handLms.landmark):
                h,w,c=img.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                

            #通过绘制对象标点连线
            mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)

    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime

    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)


    cv2.imshow('imgA',img)
    cv2.waitKey(1)