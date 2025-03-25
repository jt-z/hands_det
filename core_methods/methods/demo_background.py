import cv2
import mediapipe as mp
import numpy as np
from utils.detect_hand import extract_landmarks



# 读取模板图像
dog_template = cv2.imread('assets/dog1.webp')
rabbit_template = cv2.imread('assets/rabbit1.webp')

# 转换为灰度图像
dog_gray = cv2.cvtColor(dog_template, cv2.COLOR_BGR2GRAY)
rabbit_gray = cv2.cvtColor(rabbit_template, cv2.COLOR_BGR2GRAY)

# 应用二值化阈值分割手影和背景
_, dog_foreground_mask = cv2.threshold(dog_gray, 200, 255, cv2.THRESH_BINARY_INV)  # 200 阈值需要根据图片情况调整
_, rabbit_foreground_mask = cv2.threshold(rabbit_gray, 200, 255, cv2.THRESH_BINARY_INV)

# 显示前景掩码
cv2.imshow('Dog Foreground Mask', dog_foreground_mask)
cv2.imshow('Rabbit Foreground Mask', rabbit_foreground_mask)

# 检测手部轮廓
def detect_contours_and_extract_landmarks(image, foreground_mask):
    contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    landmarks_list = []
    
    # 遍历所有检测到的轮廓
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        hand_region = image[y:y+h, x:x+w]  # 提取每个轮廓对应的区域
        landmarks = extract_landmarks(hand_region)
        if landmarks:
            landmarks_list.append(landmarks)
    return landmarks_list

# 提取模板图像的关节点
dog_landmarks = detect_contours_and_extract_landmarks(dog_template, dog_foreground_mask)
rabbit_landmarks = detect_contours_and_extract_landmarks(rabbit_template, rabbit_foreground_mask)

# 输出关节点坐标
print("Dog Hand Landmarks:", dog_landmarks)
print("Rabbit Hand Landmarks:", rabbit_landmarks)


# 保存关节点信息（可以用于后续的匹配）
np.save('dog_landmarks.npy', dog_landmarks)
np.save('rabbit_landmarks.npy', rabbit_landmarks)

# 显示模板图片上的关节点
cv2.imshow("Dog Template", dog_template)
cv2.imshow("Rabbit Template", rabbit_template)
cv2.waitKey(0)
cv2.destroyAllWindows()

