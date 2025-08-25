"""
基于MediaPipe手部关键点的图像匹配模块
直接比较视频帧和模板原始图片之间的手部关键点相似度
"""

import cv2 as cv
import numpy as np
import mediapipe as mp
from .config import SIMILARITY_THRESHOLD
from .utils.logging_utils import get_logger, LogPerformance
from typing import Optional, Tuple, Dict, Any

class ImageMatcher:
    """基于MediaPipe关键点的图像匹配器类"""
    
    def __init__(self, default_threshold=None):
        """
        初始化图像匹配器
        
        参数:
            default_threshold (float, 可选): 默认相似度阈值，默认使用配置值
        """
        self.logger = get_logger("image_matcher")
        
        # 使用传入的阈值或默认配置
        self.default_threshold = default_threshold or SIMILARITY_THRESHOLD
        
        # 初始化MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,  # 静态图像模式
            max_num_hands=1,         # 只检测一只手
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 关键点索引定义
        self.WRIST = 0
        self.finger_tips = [4, 8, 12, 16, 20]  # 五个指尖
        self.finger_joints = {
            'thumb': [1, 2, 3, 4],
            'index': [5, 6, 7, 8], 
            'middle': [9, 10, 11, 12],
            'ring': [13, 14, 15, 16],
            'pinky': [17, 18, 19, 20]
        }
        self.finger_mcps = [2, 5, 9, 13, 17]  # 掌指关节点
        
        # 缓存，避免重复检测同一张模板图片
        self.template_landmarks_cache = {}
        
        self.logger.info(f"ImageMatcher initialized with default_threshold={self.default_threshold}")
    
    def extract_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        从图像中提取手部关键点
        
        参数:
            image (numpy.ndarray): 输入图像 (BGR格式)
            
        返回:
            numpy.ndarray: 21个关键点坐标 (21, 3) 或 None
        """
        try:
            # 转换为RGB
            rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            
            # 检测手部
            results = self.hands.process(rgb_image)
            
            if results.multi_hand_landmarks:
                # 取第一只检测到的手
                hand_landmarks = results.multi_hand_landmarks[0]
                landmarks = np.array([
                    [lm.x, lm.y, lm.z] 
                    for lm in hand_landmarks.landmark
                ])
                return landmarks
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error extracting landmarks: {e}")
            return None
    
    def normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        标准化关键点坐标 - 消除位置、尺度和旋转影响
        
        参数:
            landmarks (numpy.ndarray): 原始关键点 (21, 3)
            
        返回:
            numpy.ndarray: 标准化后的关键点
        """
        normalized = landmarks.copy()
        
        # 1. 以手腕为原点进行平移归一化
        wrist = normalized[self.WRIST]
        normalized = normalized - wrist
        
        # 2. 以手掌长度进行尺度归一化
        # 使用手腕到中指尖的距离作为特征尺度
        palm_vector = normalized[12] - normalized[0]  # 手腕到中指尖
        palm_length = np.linalg.norm(palm_vector[:2])  # 只考虑x,y平面距离
        
        if palm_length > 1e-6:
            normalized = normalized / palm_length
        
        # 3. 旋转归一化（可选）
        # 将手掌主轴对齐到y轴方向
        if palm_length > 1e-6:
            # 计算手掌主轴与y轴的夹角
            palm_angle = np.arctan2(palm_vector[0], palm_vector[1])
            
            # 构造旋转矩阵
            cos_angle = np.cos(-palm_angle)
            sin_angle = np.sin(-palm_angle)
            rotation_matrix = np.array([
                [cos_angle, -sin_angle],
                [sin_angle, cos_angle]
            ])
            
            # 对x,y坐标进行旋转
            normalized[:, :2] = np.dot(normalized[:, :2], rotation_matrix.T)
        
        return normalized
    
    def extract_gesture_features(self, landmarks: np.ndarray) -> np.ndarray:
        """
        从标准化关键点中提取手势特征向量
        
        参数:
            landmarks (numpy.ndarray): 标准化关键点 (21, 3)
            
        返回:
            numpy.ndarray: 特征向量
        """
        features = []
        
        # 1. 手指弯曲度特征 (5个特征)
        for finger_name, joints in self.finger_joints.items():
            if finger_name == 'thumb':
                # 拇指特殊处理：计算伸展程度
                tip = landmarks[joints[-1]]
                base = landmarks[joints[0]]
                extension = np.linalg.norm(tip[:2] - base[:2])  # 只考虑x,y平面
                features.append(extension)
            else:
                # 其他四指：计算关节角度总和表示弯曲程度
                bend_total = 0
                for i in range(len(joints) - 2):
                    p1 = landmarks[joints[i]][:2]
                    p2 = landmarks[joints[i+1]][:2]
                    p3 = landmarks[joints[i+2]][:2]
                    
                    # 计算关节角度
                    v1 = p1 - p2
                    v2 = p3 - p2
                    
                    dot_product = np.dot(v1, v2)
                    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
                    
                    if norms > 1e-6:
                        cos_angle = np.clip(dot_product / norms, -1, 1)
                        angle = np.arccos(cos_angle)
                        bend_total += angle
                
                features.append(bend_total)
        
        # 2. 手指间张开角度特征 (4个特征)
        wrist_pos = landmarks[self.WRIST][:2]
        for i in range(len(self.finger_tips) - 1):
            tip1 = landmarks[self.finger_tips[i]][:2] - wrist_pos
            tip2 = landmarks[self.finger_tips[i+1]][:2] - wrist_pos
            
            # 计算两指尖向量夹角
            dot_product = np.dot(tip1, tip2)
            norms = np.linalg.norm(tip1) * np.linalg.norm(tip2)
            
            if norms > 1e-6:
                cos_angle = np.clip(dot_product / norms, -1, 1)
                angle = np.arccos(cos_angle)
                features.append(angle)
            else:
                features.append(0)
        
        # 3. 手指相对位置特征 (10个特征: 5个指尖的x,y坐标)
        for tip_idx in self.finger_tips:
            tip_pos = landmarks[tip_idx][:2]
            features.extend(tip_pos)
        
        # 4. 手掌形状特征 (5个特征: 掌指关节点相对位置)
        for mcp_idx in self.finger_mcps:
            mcp_pos = landmarks[mcp_idx][:2]
            features.extend(mcp_pos)
        
        return np.array(features)
    
    def calculate_similarity_score(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """
        计算两个特征向量之间的相似度分数
        
        参数:
            features1 (numpy.ndarray): 特征向量1
            features2 (numpy.ndarray): 特征向量2
            
        返回:
            float: 相似度分数 (0-100)
        """
        try:
            # 确保特征向量长度一致
            min_len = min(len(features1), len(features2))
            features1 = features1[:min_len]
            features2 = features2[:min_len]
            
            if min_len == 0:
                return 0.0
            
            # 1. 欧几里得距离相似度 (权重: 0.3)
            euclidean_distance = np.linalg.norm(features1 - features2)
            euclidean_similarity = 1.0 / (1.0 + euclidean_distance)
            
            # 2. 余弦相似度 (权重: 0.4)
            dot_product = np.dot(features1, features2)
            norms = np.linalg.norm(features1) * np.linalg.norm(features2)
            
            if norms > 1e-6:
                cosine_similarity = dot_product / norms
                cosine_similarity = (cosine_similarity + 1) / 2  # 映射到[0,1]
            else:
                cosine_similarity = 0
            
            # 3. 皮尔逊相关系数相似度 (权重: 0.3)
            if min_len > 1:
                try:
                    correlation = np.corrcoef(features1, features2)[0, 1]
                    if np.isnan(correlation):
                        correlation = 0
                    correlation_similarity = (correlation + 1) / 2  # 映射到[0,1]
                except:
                    correlation_similarity = 0
            else:
                correlation_similarity = 0
            
            # 加权组合最终相似度
            final_similarity =   0.3 * euclidean_similarity +  0.4 * cosine_similarity +   0.3 * correlation_similarity
            
            
            # 转换为0-100分数并应用非线性增强
            score = final_similarity * 100
            
            # # 对高相似度进行增强
            # if score > 60:
            #     score = 60 + (score - 60) * 1.5  # 60分以上的相似度放大1.5倍
            
            # 确保分数在合理范围内 
            if score>100:
                score = 100 
            score = float(score)

            return score
            
        except Exception as e:
            self.logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    @LogPerformance()
    def _compare_images(self, template_image: np.ndarray, current_frame: np.ndarray) -> float:
        """
        比较模板图像和当前帧的手势相似度
        
        参数:
            template_image (numpy.ndarray): 模板图像
            current_frame (numpy.ndarray): 当前视频帧
            
        返回:
            float: 相似度分数 (0-100)
        """
        try:
            # 1. 提取模板图像关键点
            template_landmarks = self.extract_landmarks(template_image)
            if template_landmarks is None:
                self.logger.warning("Cannot extract landmarks from template image")
                return 0.0
            
            # 2. 提取当前帧关键点
            frame_landmarks = self.extract_landmarks(current_frame)
            if frame_landmarks is None:
                self.logger.warning("Cannot extract landmarks from current frame")
                return 0.0
            
            # 3. 标准化关键点
            template_normalized = self.normalize_landmarks(template_landmarks)
            frame_normalized = self.normalize_landmarks(frame_landmarks)
            
            # 4. 提取特征向量
            template_features = self.extract_gesture_features(template_normalized)
            frame_features = self.extract_gesture_features(frame_normalized)
            
            # 5. 计算相似度
            similarity = self.calculate_similarity_score(template_features, frame_features)
            
            return similarity
            
        except Exception as e:
            self.logger.error(f"Error comparing images: {e}")
            return 0.0
    
    def match_with_template(self, template_image: np.ndarray, current_frame: np.ndarray, 
                           template_name: str = "template", threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        将当前帧与模板图像进行匹配
        
        参数:
            template_image (numpy.ndarray): 模板图像
            current_frame (numpy.ndarray): 当前视频帧
            template_name (str): 模板名称
            threshold (float, 可选): 匹配阈值
            
        返回:
            dict: 匹配结果
        """
        if template_image is None or current_frame is None:
            self.logger.warning("Cannot match: template_image or current_frame is None")
            return {
                "name": template_name,
                "similarity": 0.0,
                "threshold": threshold or self.default_threshold,
                "matched": False,
                "error": "Invalid input images"
            }
        
        # 计算相似度
        similarity = self._compare_images(template_image, current_frame)
        
        # 使用指定阈值或默认阈值
        match_threshold = threshold or self.default_threshold
        
        # 构造结果
        result = {
            "name": template_name,
            "similarity": similarity,
            "threshold": match_threshold,
            "matched": similarity > match_threshold
        }
        
        self.logger.debug(f"Template match: {template_name} "
                         f"(similarity: {similarity:.2f}, "
                         f"threshold: {match_threshold}, "
                         f"matched: {result['matched']})")
        
        return result
    
    def visualize_landmarks(self, image: np.ndarray, landmarks: np.ndarray, 
                           title: str = "Hand Landmarks") -> np.ndarray:
        """
        在图像上可视化手部关键点
        
        参数:
            image (numpy.ndarray): 输入图像
            landmarks (numpy.ndarray): 关键点坐标 (21, 3)
            title (str): 窗口标题
            
        返回:
            numpy.ndarray: 带有关键点的图像
        """
        result = image.copy()
        h, w = image.shape[:2]
        
        # 绘制关键点
        for i, (x, y, z) in enumerate(landmarks):
            # 转换为像素坐标
            px, py = int(x * w), int(y * h)
            
            # 绘制关键点
            cv.circle(result, (px, py), 5, (0, 255, 0), -1)
            
            # 绘制关键点编号
            cv.putText(result, str(i), (px + 5, py - 5),
                      cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # 绘制连接线
        self._draw_hand_connections(result, landmarks, w, h)
        
        return result
    
    def _draw_hand_connections(self, image: np.ndarray, landmarks: np.ndarray, w: int, h: int):
        """绘制手部连接线"""
        # 定义连接关系
        connections = [
            # 拇指
            (0, 1), (1, 2), (2, 3), (3, 4),
            # 食指
            (0, 5), (5, 6), (6, 7), (7, 8),
            # 中指
            (0, 9), (9, 10), (10, 11), (11, 12),
            # 无名指
            (0, 13), (13, 14), (14, 15), (15, 16),
            # 小指
            (0, 17), (17, 18), (18, 19), (19, 20),
        ]
        
        for start_idx, end_idx in connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_point = (int(landmarks[start_idx][0] * w), int(landmarks[start_idx][1] * h))
                end_point = (int(landmarks[end_idx][0] * w), int(landmarks[end_idx][1] * h))
                cv.line(image, start_point, end_point, (255, 0, 0), 2)
    
    def clear_cache(self):
        """清除模板关键点缓存"""
        self.template_landmarks_cache.clear()
        self.logger.info("Template landmarks cache cleared")
    
    def __del__(self):
        """析构函数"""
        if hasattr(self, 'hands'):
            self.hands.close()