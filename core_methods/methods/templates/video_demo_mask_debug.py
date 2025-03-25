# 用模板方法，显示很多中间结果。

import sys
import os


print("PYTHONPATH:", os.environ.get('PYTHONPATH')) 
print("Current working directory:", os.getcwd())
print("Python path:", sys.path)

# 手动添加项目根目录到sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, project_root)

print("Updated Python path:", sys.path)
  
import cv2
import numpy as np
import os
import mediapipe as mp
from scipy.spatial import distance
from utils.detect_hand import extract_hand,extract_landmarks

# 初始化MediaPipe的手部检测模块
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

 
# 加载手影模板
dog_template_path = '../../assets/mask_dog.png'
rabbit_template_path = '../../assets/mask_rabit.png'

# 读取并预处理模板图像
dog_template = cv2.imread(dog_template_path, cv2.IMREAD_GRAYSCALE)
rabbit_template = cv2.imread(rabbit_template_path, cv2.IMREAD_GRAYSCALE)

# 预处理模板（灰度化和二值化）
_, dog_template = cv2.threshold(dog_template, 127, 255, cv2.THRESH_BINARY)
_, rabbit_template = cv2.threshold(rabbit_template, 127, 255, cv2.THRESH_BINARY)

# 图像哈希函数
def hash_image(img):
    resized = cv2.resize(img, (270, 180))  # 缩小图像
    avg_color = resized.mean()
    hashed = (resized > avg_color).astype(np.uint8)
    return hashed.flatten()

# 哈希比较函数
def compare_hashes(hash1, hash2):
    return np.sum(hash1 != hash2)

# 模板哈希
dog_template_hash = hash_image(dog_template)
rabbit_template_hash = hash_image(rabbit_template)


video_file = '../../assets/wolf_and_chicken.mp4'  # 替换为你的视频文件路径

# 读取本地视频文件
cap = cv2.VideoCapture(video_file)

print(f'Reading video: {video_file}')

frame_count = 0  # 初始化帧计数器

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1  # 增加帧计数器
    
    # 转换视频帧为灰度图并进行预处理
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray Frame', gray_frame)  # 显示灰度图
    print('Gray Frame Shape:', gray_frame.shape)  # 打印灰度图形状

    _, binary_frame = cv2.threshold(gray_frame, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow('Binary Frame', binary_frame)  # 显示二值图
    print('Binary Frame Thresholded')  # 提示二值化成功
    
    # 检测轮廓并找到最大的轮廓
    contours, _ = cv2.findContours(binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        print('Largest Contour Area:', cv2.contourArea(largest_contour))  # 输出最大轮廓的面积
        
        if cv2.contourArea(largest_contour) > 1000:  # 忽略小轮廓
            # 创建轮廓掩码
            mask = np.zeros_like(binary_frame)
            cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
            cv2.imshow('Mask', mask)  # 显示轮廓掩码
            print('Mask Created')  # 提示掩码创建成功
            
            # 缩放到和模板相同的大小
            mask_resized = cv2.resize(mask, (100, 100))
            cv2.imshow('Resized Mask', mask_resized)  # 显示缩放后的掩码
            print('Resized Mask Shape:', mask_resized.shape)  # 输出缩放后掩码的形状
            
            # 计算当前帧的哈希
            current_frame_hash = hash_image(mask_resized)
            
            # 与模板进行比较
            dog_similarity = compare_hashes(current_frame_hash, dog_template_hash)
            rabbit_similarity = compare_hashes(current_frame_hash, rabbit_template_hash)
            
            # 输出与模板的相似度
            print(f'Dog Similarity: {dog_similarity}, Rabbit Similarity: {rabbit_similarity}')
            

            # 找出最相似的模板
            if dog_similarity < rabbit_similarity:
                label = 'Dog Hand Shadow'
            else:
                label = 'Rabbit Hand Shadow'
            
            # 在视频中显示匹配结果
            cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            print(f'Detected: {label}')  # 打印检测结果
    
    # 每处理10帧，等待按键输入
    if frame_count % 10 == 0:
        print(f"Processed {frame_count} frames. Press any key to continue.")
        cv2.waitKey(0)  # 等待用户按键后继续
    
    # 显示视频
    cv2.imshow("Hand Shadow Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
