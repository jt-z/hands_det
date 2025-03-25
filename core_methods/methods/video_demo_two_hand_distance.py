import cv2
import numpy as np
from scipy.spatial import distance
from utils.detect_hand import extract_landmarks

# 加载保存的模板关节点信息（确保模板数据存储的是两只手的关节点）
dog_landmarks_template = np.load('assets/dog_landmarks.npy', allow_pickle=True)
rabbit_landmarks_template = np.load('assets/rabbit_landmarks.npy', allow_pickle=True)

# 函数：计算欧氏距离，比较关节点与模板的相似度
def compare_landmarks(landmarks, template_landmarks):
    if len(landmarks) != len(template_landmarks):
        return float('inf')  # 如果关节点数量不一致，返回无穷大距离
    return np.mean([distance.euclidean(landmark, template) for landmark, template in zip(landmarks, template_landmarks)])

# 函数：比较两只手的关节点与模板，并计算平均相似度
def compare_two_hands(landmarks, template):
    if len(landmarks) != 2 or len(template) != 2:
        return float('inf')  # 如果检测不到两只手，返回无穷大距离
    hand1_similarity = compare_landmarks(landmarks[0], template[0])
    hand2_similarity = compare_landmarks(landmarks[1], template[1])
    return (hand1_similarity + hand2_similarity) / 2  # 返回两只手的平均相似度

# 函数：计算两只手之间的距离，使用手腕或手心作为参考点
def hands_distance(landmarks):
    hand1_center = landmarks[0][0]  # 使用手腕（关节点索引 0）
    hand2_center = landmarks[1][0]  # 使用另一只手的手腕
    return distance.euclidean(hand1_center, hand2_center)

# 函数：根据距离和模板匹配结果，返回识别的手影类型
def detect_hand_shadow(current_landmarks, distance_threshold=0.2):
    hands_dist = hands_distance(current_landmarks)
    
    # 如果两只手的距离太远，跳过匹配
    if hands_dist > distance_threshold:
        return 'Hands too far apart, closeer!'
    
    # 比较与狗手影模板的相似度
    dog_similarity = compare_two_hands(current_landmarks, dog_landmarks_template)
    
    # 比较与兔子手影模板的相似度
    rabbit_similarity = compare_two_hands(current_landmarks, rabbit_landmarks_template)
    
    # 根据相似度输出匹配结果
    if dog_similarity < rabbit_similarity:
        return 'bird Hand Shadow'
    else:
        return 'wolf Hand Shadow'

# 主函数：运行整个手影识别过程
def main():

    video_file = 'assets/wolf_and_chicken.mp4'  # 替换为你的视频文件路径
    video_file = 'assets/self_get.mov'  # 替换为你的视频文件路径

    # 启动摄像头捕获视频
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(video_file)


    # 获取视频的宽度、高度和帧率
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # 初始化视频写入器
    out = cv2.VideoWriter('output_video2.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width, frame_height))
    
    
    # 设置一个距离阈值，确保两只手必须接近
    distance_threshold = 0.2  # 需要根据图像坐标值来调整
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 提取视频中实时的手部关节点
        current_landmarks = extract_landmarks(frame)
        
        if current_landmarks and len(current_landmarks) == 2:  # 确保检测到两只手
            # 获取手影类型标签
            label = detect_hand_shadow(current_landmarks, distance_threshold)
            
            # 在视频中显示匹配结果
            cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            # 如果检测不到两只手，显示错误信息
            cv2.putText(frame, 'Make sure has two hands.', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # 显示视频
        cv2.imshow("Hand Shadow Detection", frame)

                
        # 写入处理后的帧到输出视频
        out.write(frame)
        
        # 按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# 入口：调用主函数
if __name__ == "__main__":
    main()
