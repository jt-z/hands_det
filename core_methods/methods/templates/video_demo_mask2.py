import cv2
import numpy as np
import os 
from scipy.spatial import distance

import sys
# 手动添加项目根目录到sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, project_root)

from utils.detect_hand import extract_hand
 



# 图像哈希函数，用于模板匹配
def hash_image(img, size=(270, 180)):
    resized = cv2.resize(img, size)
    avg_color = resized.mean()
    hashed = (resized > avg_color).astype(np.uint8)
    return hashed.flatten()

# 哈希比较函数，返回哈希之间的差异度
def compare_hashes(hash1, hash2):
    return np.sum(hash1 != hash2)

# 加载并预处理手影模板
def load_and_preprocess_templates(dog_template_path, rabbit_template_path):
    dog_template = cv2.imread(dog_template_path, cv2.IMREAD_GRAYSCALE)
    rabbit_template = cv2.imread(rabbit_template_path, cv2.IMREAD_GRAYSCALE)
    _, dog_template = cv2.threshold(dog_template, 127, 255, cv2.THRESH_BINARY)
    _, rabbit_template = cv2.threshold(rabbit_template, 127, 255, cv2.THRESH_BINARY)
    dog_template_hash = hash_image(dog_template)
    rabbit_template_hash = hash_image(rabbit_template)
    return dog_template_hash, rabbit_template_hash

# 处理视频帧并检测手影轮廓
def process_frame(frame, dog_template_hash, rabbit_template_hash):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary_frame = cv2.threshold(gray_frame, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 1000:
            mask = np.zeros_like(binary_frame)
            cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
            mask_resized = cv2.resize(mask, (100, 100))
            current_frame_hash = hash_image(mask_resized)
            dog_similarity = compare_hashes(current_frame_hash, dog_template_hash)
            rabbit_similarity = compare_hashes(current_frame_hash, rabbit_template_hash)
            label = 'Dog Hand Shadow' if dog_similarity < rabbit_similarity else 'Rabbit Hand Shadow'
            cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame

# 读取本地视频文件并进行手影识别
def detect_hand_shadows_in_video(video_file, dog_template_hash, rabbit_template_hash):
    cap = cv2.VideoCapture(video_file)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
 
        frame, hand_bboxes, hand_landmarks_list = extract_hand(frame)
        # if len(hand_landmarks_list)>0:
        #    for hand_landmarks in hand_landmarks_list:
        #        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        #    for bbox in hand_bboxes:
        #        cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

        frame = process_frame(frame, dog_template_hash, rabbit_template_hash)


        cv2.imshow("Hand Shadow Detection", frame)

                
        if frame_count % 1000 == 0:
            print(f"Processed {frame_count} frames. Press any key to continue.")
            cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# 主函数入口
if __name__ == "__main__":
    dog_template_path = '../../assets/mask_dog.png'
    rabbit_template_path = '../../assets/mask_rabit.png'
    video_file = '../../assets/wolf_and_chicken.mp4'
    
    # 加载并预处理模板
    dog_template_hash, rabbit_template_hash = load_and_preprocess_templates(dog_template_path, rabbit_template_path)

    # 处理视频文件并检测手影
    detect_hand_shadows_in_video(video_file, dog_template_hash, rabbit_template_hash)
