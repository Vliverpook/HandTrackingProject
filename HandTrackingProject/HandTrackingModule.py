#功能模块化
import cv2
import mediapipe as mp
import time

#创建检测类
class handDetector():
    #参数初始化
    def __init__(self,mode=False,maxHands=2,detectionCon=0.5,trackCon=0.5):
        self.mode=mode
        self.maxHands=maxHands
        self.detectionCon=detectionCon
        self.trackCon=trackCon



        # 新建手部检测对象
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,1,self.detectionCon,self.trackCon)

        # 新建绘点对象
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self,img,draw=True):
        # 将BGR转为RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 传入手部图片进行检测
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            # 每组检测结果可能识别出多个手
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    # 通过绘制对象标点连线
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    #通过点的序号找到所有检测到的手中特定点的位置
    def findPosition(self,img,handNo=0,draw=True):
        lmList=[]

        if self.results.multi_hand_landmarks:
            myHand=self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)

        return lmList


# cap = cv2.VideoCapture(0)
#
# # 新建手部检测对象
# mpHands = mp.solutions.hands
# hands = mpHands.Hands()
#
# # 新建绘点对象
# mpDraw = mp.solutions.drawing_utils
#
# # 用来计算帧率
# pTime = 0
# cTime = 0
#
# while True:
#     success, img = cap.read()
#
#     # 将BGR转为RGB
#     imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#     # 传入手部图片进行检测
#     results = hands.process(imgRGB)
#
#     # print(results.multi_hand_landmarks)
#
#     # 当有检测结果时
#     if results.multi_hand_landmarks:
#         # 每组检测结果可能识别出多个手
#         for handLms in results.multi_hand_landmarks:
#             # 获取检测结果点的xy坐标
#             for id, lm in enumerate(handLms.landmark):
#                 h, w, c = img.shape
#                 cx, cy = int(lm.x * w), int(lm.y * h)
#
#             # 通过绘制对象标点连线
#             mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
#
#     cTime = time.time()
#     fps = 1 / (cTime - pTime)
#     pTime = cTime
#
#     cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
#
#     cv2.imshow('imgA', img)
#     cv2.waitKey(1)

def main():
    # 用来计算帧率
    pTime = 0
    cTime = 0
    #初始化对象
    detector=handDetector()
    cap =cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        img=detector.findHands(img)
        lmList=detector.findPosition(img)
        if len(lmList)!=0:
            print(lmList[4])
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow('imgA', img)
        cv2.waitKey(1)

if __name__=='__main__':
    main()