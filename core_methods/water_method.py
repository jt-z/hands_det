#分水岭算法 效果

import cv2
import numpy as np
import matplotlib.pyplot as plt
 
img = cv2.imread('assets/hand.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
plt.hist(gray.ravel(),bins=256,range=[0,255])  #用plt画直方图，ravel()函数是把多维变一维
plt.show()
 
#img的图是典型的双峰结构，用大津算法进行二值化处理
_,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)  #全局二值化，大津算法自动找阈值
 
#二值化后的图存在毛边，有小噪点，做一下开运算
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)) #结构元
img_open= cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=2)
# cv2.imshow('img_open',img_open)
 
#想办法找到前景和背景
#对img_open进行膨胀操作，找背景
bg = cv2.dilate(img_open,kernel,iterations=2)
cv2.imshow('bg',bg)
 
#对img_open进行腐蚀操作，找前景,但是从前景图上来看，效果不太好，因为硬币与硬币之间有明显的通道，跟实际（相切）不一样
# fg = cv2.erode(img_open,kernel,iterations=2)
# cv2.imshow('fg',fg)
 
#可以通过膨胀减去腐蚀，就是硬币的边界，即未知区域
# unkown = cv2.subtract(bg,fg)
# cv2.imshow('unkown',unkown)
 
#通过腐蚀来确定前景不合适，用distanceTransform()来确定前景
dist_fg = cv2.distanceTransform(img_open,cv2.DIST_L2, 5)
#对dist_fg做归一化方便展示结果
dist_fg = cv2.normalize(dist_fg,None,0,1.0,cv2.NORM_MINMAX)
# print('dist_fg.max:',dist_fg.max())
cv2.imshow('dist_fg',dist_fg)
 
#对dist_fg做二值化处理
_,fg = cv2.threshold(dist_fg,0.6*dist_fg.max(),255,cv2.THRESH_BINARY)
cv2.imshow('fg',fg)
fg = np.uint8(fg)  #把fg的数据类型转换位uint8的整型
# print(fg)
unkown = cv2.subtract(bg,fg)  #计算未知区域，硬币边缘
cv2.imshow('unkown',unkown)
 
#connectedComponents要求输入的图片是8位的单通道图片，单通道的值是0-255的整型。这个函数可以计算出标志区域（0标记背景，大于0的整数标记前景）
_,markers = cv2.connectedComponents(fg)
print('markers_max:',markers.max(),'markers_min:',markers.min())  #marks大小和输入图片一样
 
#因为分水岭算法watershed中是：0是未知区域，1是背景，大于1是前景，markers +1的话，把原来的0变为1即可。
markers += 1
#从markers中筛选出未知区域，然后赋值位0
markers[unkown == 255] = 0  #此时watershed需要的markers已经完成
print(markers.max())
 
#展示一下markers
markers_copy = markers.copy()
markers_copy[markers == 0] =127 #位置区域
markers_copy[markers >1] = 255  #前景
markers_copy = markers_copy.astype('uint8')  #要注意需要把其类型转换位uint8才能展示图片
cv2.imshow('markers_copy',markers_copy)
 
#执行分水岭算法
markers = cv2.watershed(img,markers)  #返回的markers做了修改，边界区域标记为了-1
print('markers:',markers.max(),markers.min())
 
# img[markers == -1] = [0,0,255]  #标记边缘
# # cv2.imshow('img',img)
 
# img[markers > 1] = [0,255,0]  #标记前景
# cv2.imshow('img',img)
 
#抠出硬币
#mask把要抠图的地方赋值为255，其他位置赋值为0
mask = np.zeros(shape=img.shape[:2],dtype=np.uint8)
mask[markers > 1] = 255
img_coins = cv2.bitwise_and(img,img,mask=mask)
cv2.imshow('img_coins',img_coins)
 
cv2.waitKey(0)
cv2.destroyAllWindows()