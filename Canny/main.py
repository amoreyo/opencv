# 我为什么要用python实现Canny代码？因为懒，因为python写起来相对无脑（当然时间花费++），因为之前用python写过sobel算子还有高斯模糊
# 这里主要实现的是Canny中非极大值抑制和双阈值

import numba as nb
import numpy as np
from PIL import Image
import math
import cv2

stride = 1
kernel_size = 3
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

def conv(grey_array):
    # 从每一行这样开始遍历走stride吧
    conv_array = np.zeros((grey_array.shape[0], grey_array.shape[1]),dtype=float)
    # padding 肯定会padding的呀
    # if(padding):
    #     padding_size = int((kernel_size-1)/2)
    # else:
    #     padding_size = 0
    for i in range(grey_array.shape[0]-2):
        for j in range(grey_array.shape[1]-2):
            conv_array[i,j] += (grey_array[i,j]*Sx[0,0] + grey_array[i,j+1]*Sx[0,1] + grey_array[i,j+2]*Sx[0,2])/3
            conv_array[i,j] += (grey_array[i+1,j]*Sx[1,0] + grey_array[i+1,j+1]*Sx[1,1] + grey_array[i+1,j+2]*Sx[1,2])/3
            conv_array[i,j] += (grey_array[i + 2, j] * Sx[2, 0] + grey_array[i + 2, j + 1] * Sx[2, 1] + grey_array[i + 2, j + 2] * Sx[2, 2])/3

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
    conv_array_x = conv(grey_array)
    conv_array_y = conv(grey_array)

    grad_array = grad(conv_array_x, conv_array_y)
    # amp_array = (conv_array_x**2 + conv_array_y**2)**0.5
    return conv_array_x, conv_array_y, grad_array# , amp_array


def NMS():
    return 0

img = cv2.imread("img/Q3ca8sqEgB.jpg")
imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

conv_array_x, conv_array_y, grad_array = sobel_xy(imgGrey)
print(grad_array)
print(conv_array_x[0][0])
print(conv_array_y[0][0])
# 先明确我们需要的参数
# x,y轴的偏导，记为E(x),E(y)
# 幅值，记为M
# 梯度，记为0

