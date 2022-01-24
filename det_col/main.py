import cv2
import numpy as np
# cap = cv2.VideoCapture(0)
# cap.set(3,360) # 3 is the size
# cap.set(10,200) # 10 is the bright
#
# while True:
#     success, img = cap.read()
#     cv2.imshow("video", img)
#     # press Q and the video stop
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# img = cv2.imread("img/Q3ca8sqEgB.jpg") #BGR
# print(img.shape) # (1350, 1080, 3) x,y,c
# imgResize = cv2.resize(img,(600, 200))
#
# imgCropped = imgResize[0:500, 100:200]
# cv2.imshow("Resize",imgResize)
# cv2.imshow("Cropped", imgCropped)
# cv2.waitKey(0)

# kernel = np.ones((5,5),np.uint8)
# imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# imgBlur = cv2.GaussianBlur(imgGrey,ksize=(7,7),sigmaX=0)
# imgCanny = cv2.Canny(imgGrey,threshold1=100,threshold2=100)
# # 膨胀
# imgDialation = cv2.dilate(imgCanny, kernel, iterations=1)
# # 侵蚀
# imgEroded = cv2.erode(imgDialation,kernel,iterations=1)
# cv2.imshow("Grey", imgGrey)
# cv2.imshow("Blur", imgBlur)
# cv2.imshow("Cannt", imgCanny)
# cv2.imshow("Dialation", imgDialation)
# cv2.imshow("Eroded", imgEroded)
# cv2.waitKey(2000)

# img = np.zeros((512, 512, 3),np.uint8)
# print(img.shape)
# img[:] = 255,0,0
# cv2.line(img,(0,0),(img.shape[1],img.shape[0]),(0,255,0),3)
# cv2.rectangle(img,(0,0),(250,350),(255,0,255),cv2.FILLED)
# cv2.circle(img,(400,50),30,(255,255,255),10)
# cv2.putText(img,"opencv",(110,330),cv2.FONT_HERSHEY_SIMPLEX,2,(0,123,0),5)


# img = cv2.imread("img/pock.png")
# width, height = 250,350
# pts1 = np.float32([[55,125],[157,124],[10,262],[135,260]])
# pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
# matrix = cv2.getPerspectiveTransform(pts1,pts2)
# imgOutput = cv2.warpPerspective(img,matrix,(width,height))
# cv2.imshow("img",img)
# cv2.imshow("output",imgOutput)

# img = cv2.imread("img/Q3ca8sqEgB.jpg")
# img = cv2.resize(img,(200,200))
# hor = np.hstack((img,img))
# ver = np.vstack((img,img))
# vers = np.vstack((ver,img))
# cv2.imshow("Ver",ver)
# cv2.imshow("Vers",vers)
# cv2.imshow("Hor",hor)



cv2.waitKey(0)