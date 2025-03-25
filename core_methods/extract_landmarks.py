# 只能检测到一个，或者是容易检测到手影，需要考虑调整。

# 查看mediapipe的检测效果，检测两张图片试试。

import cv2
import mediapipe as mp
import numpy as np
from utils.detect_hand import extract_landmarks


# 读取模板图像
dog_template = cv2.imread('assets/dog1.webp')
rabbit_template = cv2.imread('assets/rabbit1.webp')

# 提取模板图像的关节点
dog_landmarks = extract_landmarks(dog_template)
rabbit_landmarks = extract_landmarks(rabbit_template)

# 输出关节点坐标
print("Dog Hand Landmarks:", dog_landmarks)
print("Rabbit Hand Landmarks:", rabbit_landmarks)

# 显示模板图片上的关节点
cv2.imshow("Dog Template", dog_template)
cv2.imshow("Rabbit Template", rabbit_template)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存关节点信息（可以用于后续的匹配）
np.save('assets/dog_landmarks.npy', dog_landmarks)
np.save('assets/rabbit_landmarks.npy', rabbit_landmarks)
