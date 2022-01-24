# 我为什么要用python实现Canny代码？因为懒，因为python写起来相对无脑（当然时间花费++），因为之前用python写过sobel算子还有高斯模糊
# 这里主要实现的是Canny中非极大值抑制和双阈值
# 在cv2.imshow中为什么明明灰度图最大值是55，却显示出来像255？？？？？？？
# 可能是我dtype=float的原因？
# 不是
# 双阈值算法
# （1） 根据图像选取合适的高阈值和低阈值，通常高阈值是低阈值的2到3倍
# （2） 如果某一像素的梯度值高于高阈值，则保留
# （3） 如果某一像素的梯度值低于低阈值，则舍弃
# （4） 如果某一像素的梯度值介于高低阈值之间，则从该像素的8邻域的寻找像素梯度值，如果存在像素梯度值高于高阈值，则保留，如果没有，则舍弃
# 先遍历一遍高阈值，再执行4步骤
import numpy as np
import math
import cv2

stride = 1
kernel_size = 3
beta = 0.5
tao_h = 0.7
tao_l = 0.4

# sobel算子
Sx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
Sy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
# 听说是更好的算子
Sx_up = np.array([[-3,0,3],[-10,0,10],[-3,0,3]])
Sy_up = np.array([[3,10,3],[0,0,0],[-3,-10,-3]])

def grad(conv_array_x, conv_array_y):
    grad_array = np.zeros((conv_array_x.shape[0], conv_array_x.shape[1]), dtype=float)
    for i in range(conv_array_x.shape[0]):
        for j in range(conv_array_x.shape[1]):
            if(conv_array_y[i, j] == 0):
                grad_array[i, j] = math.pi/2
            else:
                grad_array[i, j] = np.arctan(conv_array_x[i, j] / conv_array_y[i, j])
    return grad_array

def conv(grey_array, xy):
    # 从每一行这样开始遍历走stride吧
    max = 0
    conv_array = np.zeros((grey_array.shape[0]-2, grey_array.shape[1]-2),dtype=float)
    if xy == "x":
        for i in range(grey_array.shape[0]-2):
            for j in range(grey_array.shape[1]-2):
                conv_array[i,j] += (grey_array[i,j]*Sx[0,0] + grey_array[i,j+1]*Sx[0,1] + grey_array[i,j+2]*Sx[0,2])/8
                conv_array[i,j] += (grey_array[i+1,j]*Sx[1,0] + grey_array[i+1,j+1]*Sx[1,1] + grey_array[i+1,j+2]*Sx[1,2])/8
                conv_array[i,j] += (grey_array[i + 2, j] * Sx[2, 0] + grey_array[i + 2, j + 1] * Sx[2, 1] + grey_array[i + 2, j + 2] * Sx[2, 2])/8
                # print(conv_array[i,j] , grey_array[i+1,j+1])
                # if conv_array[i,j] > max:
                #     max = conv_array[i,j]
                #     print(max)
                    # max = 63.125
    else:
        for i in range(grey_array.shape[0]-2):
            for j in range(grey_array.shape[1]-2):
                conv_array[i,j] += (grey_array[i,j]*Sy[0,0] + grey_array[i+1,j]*Sy[1, 0] + grey_array[i+2,j]*Sy[2, 0])/8
                conv_array[i,j] += (grey_array[i,j+1]*Sy[0,1] + grey_array[i+1,j+1]*Sy[1,1] + grey_array[i+2,j+1]*Sy[2,1])/8
                conv_array[i,j] += (grey_array[i,j+2]*Sy[0,2] + grey_array[i+1,j+2]*Sy[1,2] + grey_array[i+2,j+2]*Sy[2,2])/8
                # if conv_array[i,j] > max:
                #     max = conv_array[i,j]
                #     print(max)
                #     max = 55.25
                # print(conv_array[i, j] , grey_array[i+1,j+1])
    # for i in range(10):
    #     conv_array[50+i] = 255
    # conv_array = np.uint8(conv_array)
    return conv_array

def sobel_xy(grey_array):
    # if(resize):
    #     # pad_array = pad(grey_array, kernel_size)
    #     # 没错我写的padding太费时间拉，倒不如直接cv2.resize嘿嘿
    #     pad_array = cv2.resize(grey_array, (130,130))
    # else:
    #     pad_array = grey_array
    # steps = grey_array.h * w
    # 返回一个np数组，存x方向上的梯度
    conv_array_x = conv(grey_array, 'x')
    conv_array_y = conv(grey_array, 'y')

    grad_array = grad(conv_array_x, conv_array_y)
    amp_array = (conv_array_x**2 + conv_array_y**2)**0.5
    return conv_array_x, conv_array_y, grad_array, amp_array

# 放弃了，太烦了
def NMS(imgBlur, conv_array_x, conv_array_y, grad_array, amp_array):
    nms_array = np.zeros((imgBlur.shape[0] - 2, imgBlur.shape[1] - 2), dtype=float)
# [-pi/2 ~ -pi/4) ; [-pi/4 ~ 0) ; [0 ~ pi/4) ; [pi/4 ~ pi/2]
    for i in range(conv_array_x.shape[0]):
        for j in range(conv_array_x.shape[1]):
            # K(up) = I(i-1, j) + beta[I(i-1, j+1) - I(i-1, j)]
            # K(down)
            # M = K(down) + beta * Ey/Ex *[K(up) - K(down)]
            if grad_array[i][j] >= -math.pi/2 and grad_array[i][j] < -math.pi/4:
                Kup1 = imgBlur[i][j] + beta * (imgBlur[i][j+1] - imgBlur[i][j])
                Kdown1 = imgBlur[i+1][j] + beta * (imgBlur[i+1][j+1] - imgBlur[i+1][j])
                M1 = math.fabs(Kup1 - Kdown1)
                Kup2 = imgBlur[i+1][j+1] + beta * (imgBlur[i+1][i+2] - imgBlur[i+1][j+1])
                Kdown2 = imgBlur[i + 2][j+1] + beta * (imgBlur[i + 2][j + 2] - imgBlur[i + 2][j+1])
                M2 = math.fabs(Kup2 - Kdown2)
                if amp_array[i,j] > M2 and amp_array[i,j] > M1:
                    nms_array[i,j] = amp_array[i,j]
            elif grad_array[i][j] >= -math.pi/4 and grad_array[i][j] < 0:
                Kup1 = imgBlur[i][j] + beta * (imgBlur[i][j+1] - imgBlur[i][j])
                Kdown1 = imgBlur[i+1][j] + beta * (imgBlur[i+1][j+1] - imgBlur[i+1][j])
                M1 = math.fabs(Kup1 - Kdown1)
                Kup2 = imgBlur[i+1][j+1] + beta * (imgBlur[i+1][i+2] - imgBlur[i+1][j+1])
                Kdown2 = imgBlur[i + 2][j+1] + beta * (imgBlur[i + 2][j + 2] - imgBlur[i + 2][j+1])
                M2 = math.fabs(Kup2 - Kdown2)
                if amp_array[i,j] > M2 and amp_array[i,j] > M1:
                    nms_array[i,j] = amp_array[i,j]
            elif grad_array[i][j] >= 0 and grad_array[i][j] < math.pi/4:
                Kup1 = imgBlur[i][j] + beta * (imgBlur[i][j+1] - imgBlur[i][j])
                Kdown1 = imgBlur[i+1][j] + beta * (imgBlur[i+1][j+1] - imgBlur[i+1][j])
                M1 = math.fabs(Kup1 - Kdown1)
                Kup2 = imgBlur[i+1][j+1] + beta * (imgBlur[i+1][i+2] - imgBlur[i+1][j+1])
                Kdown2 = imgBlur[i + 2][j+1] + beta * (imgBlur[i + 2][j + 2] - imgBlur[i + 2][j+1])
                M2 = math.fabs(Kup2 - Kdown2)
                if amp_array[i,j] > M2 and amp_array[i,j] > M1:
                    nms_array[i,j] = amp_array[i,j]
            elif grad_array[i][j] >= math.pi / 4 and grad_array[i][j] <= math.pi/2:
                Kup1 = imgBlur[i][j] + beta * (imgBlur[i][j+1] - imgBlur[i][j])
                Kdown1 = imgBlur[i+1][j] + beta * (imgBlur[i+1][j+1] - imgBlur[i+1][j])
                M1 = math.fabs(Kup1 - Kdown1)
                Kup2 = imgBlur[i+1][j+1] + beta * (imgBlur[i+1][i+2] - imgBlur[i+1][j+1])
                Kdown2 = imgBlur[i + 2][j+1] + beta * (imgBlur[i + 2][j + 2] - imgBlur[i + 2][j+1])
                M2 = math.fabs(Kup2 - Kdown2)
                if amp_array[i,j] > M2 and amp_array[i,j] > M1:
                    nms_array[i,j] = amp_array[i,j]

    return 0


# array = np.zeros((255,255),dtype=np.uint8)
# for i in range(255):
#     for j in range(255):
#         array[i,j] = 255
# cv2.imshow("name",array)
# print(math.fabs(-1))
img = cv2.imread("img/CZxxWmy0oU.jpg")
imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGrey,ksize=(3,3),sigmaX=0)

conv_array_x, conv_array_y, grad_array, amp_array = sobel_xy(imgBlur)
# cv2.imshow("img",amp_array)
cv2.imshow("img0", imgBlur)
cv2.imshow("img", conv_array_x)
cv2.imshow("img1", conv_array_y)
cv2.waitKey(0)
# NMS(imgBlur, conv_array_x, conv_array_y, grad_array, amp_array)

# cv2.imshow("img", conv_array_y)
# cv2.imshow("img2",conv_array_x)
# cv2.imshow("grad", grad_array)
# cv2.waitKey(0)

# 先明确我们需要的参数
# x,y轴的偏导，记为E(x),E(y)
# 幅值，记为M
# 梯度，记为0

# for i in range(grad_array.shape[0]):
#     for j in range(len(grad_array[0])):
#         print(grad_array[i][j])
# so the grad_array's element's range is -pi/2 ~ pi/2
#                                        -90   ~  90
#                                        4 dim each is [-pi/2 ~ -pi/4) ; [-pi/4 ~ 0) ; [0 ~ pi/4) ; [pi/4 ~ pi/2]
# K(up) = I(i-1, j) + beta[I(i-1, j+1) - I(i-1, j)]
# K(down)
# M = K(down) + beta * Ey/Ex *[K(up) - K(down)]


