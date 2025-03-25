import cv2
import mediapipe as mp

# 初始化MediaPipe的手部检测模块
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 提取手部关节点并返回处理过的图像、边界框坐标以及关节点坐标
def extract_hand(image, margin=20):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.1) as hands:
        results = hands.process(image_rgb)
        hand_bboxes = []
        hand_landmarks_list = []
        if results.multi_hand_landmarks:
            h, w, _ = image.shape
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
                hand_landmarks_list.append(landmarks)
                min_x = min(landmarks, key=lambda x: x[0])[0] - margin
                min_y = min(landmarks, key=lambda x: x[1])[1] - margin
                max_x = max(landmarks, key=lambda x: x[0])[0] + margin
                max_y = max(landmarks, key=lambda x: x[1])[1] + margin
                min_x, min_y = max(0, min_x), max(0, min_y)
                max_x, max_y = min(w, max_x), min(h, max_y)
                hand_bboxes.append(((min_x, min_y), (max_x, max_y)))
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
        return image, hand_bboxes, hand_landmarks_list

# 函数：提取手部关节点并返回关节点坐标
def extract_landmarks(image):
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.1) as hands:  # 调整检测置信度
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        landmarks_list = []  # 用于存储所有手的关节点
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
                landmarks_list.append(landmarks)  # 将每只手的关节点都存储进列表
        return landmarks_list if landmarks_list else None