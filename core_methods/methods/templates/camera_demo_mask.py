import cv2
import numpy as np
import os

# 加载手影模板
dog_template_path = 'assets/mask_dog.png'
rabbit_template_path = 'assets/mask_rabit.png'

# 读取并预处理模板图像
dog_template = cv2.imread(dog_template_path, cv2.IMREAD_GRAYSCALE)
rabbit_template = cv2.imread(rabbit_template_path, cv2.IMREAD_GRAYSCALE)

# 预处理模板（灰度化和二值化）
_, dog_template = cv2.threshold(dog_template, 127, 255, cv2.THRESH_BINARY)
_, rabbit_template = cv2.threshold(rabbit_template, 127, 255, cv2.THRESH_BINARY)

# 图像哈希函数
def hash_image(img):
    resized = cv2.resize(img, (30, 30))  # 缩小图像
    avg_color = resized.mean()
    hashed = (resized > avg_color).astype(np.uint8)
    return hashed.flatten()

# 哈希比较函数
def compare_hashes(hash1, hash2):
    return np.sum(hash1 != hash2)

# 模板哈希
dog_template_hash = hash_image(dog_template)
rabbit_template_hash = hash_image(rabbit_template)


camera_id = 0
# 启动摄像头捕获视频
cap = cv2.VideoCapture(camera_id)

print(f'use camera {camera_id}')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 转换视频帧为灰度图并进行预处理
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary_frame = cv2.threshold(gray_frame, 127, 255, cv2.THRESH_BINARY)
    
    # 检测轮廓并找到最大的轮廓
    contours, _ = cv2.findContours(binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 1000:  # 忽略小轮廓
            # 创建轮廓掩码
            mask = np.zeros_like(binary_frame)
            cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
            
            # 缩放到和模板相同的大小
            mask_resized = cv2.resize(mask, (100, 100))
            
            # 计算当前帧的哈希
            current_frame_hash = hash_image(mask_resized)
            
            # 与模板进行比较
            dog_similarity = compare_hashes(current_frame_hash, dog_template_hash)
            rabbit_similarity = compare_hashes(current_frame_hash, rabbit_template_hash)
            
            # 找出最相似的模板
            if dog_similarity < rabbit_similarity:
                label = 'Dog Hand Shadow'
            else:
                label = 'Rabbit Hand Shadow'
            
            # 在视频中显示匹配结果
            cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 显示视频
    cv2.imshow("Hand Shadow Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
