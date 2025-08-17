import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Tuple, Optional, List, Union
import os
import json
from dataclasses import dataclass
from pathlib import Path
from enum import Enum


class HandType(Enum):
    """手的类型"""
    LEFT = "Left"
    RIGHT = "Right"
    BOTH = "Both"


@dataclass
class GestureTemplate:
    """手势模板数据结构"""
    name: str
    hand_type: HandType
    left_landmarks: Optional[np.ndarray]
    right_landmarks: Optional[np.ndarray]
    left_features: Optional[np.ndarray]
    right_features: Optional[np.ndarray]
    combined_features: np.ndarray
    image_path: str


class TwoHandGestureRecognizer:
    """双手手势识别与相似度匹配系统"""
    
    def __init__(self, templates_dir: str = "templates"):
        """
        初始化双手手势识别器
        
        Args:
            templates_dir: 模板图片存储目录
        """
        # MediaPipe初始化
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,  # 检测两只手
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 模板存储
        self.templates: Dict[str, GestureTemplate] = {}
        self.templates_dir = Path(templates_dir)
        
        # 关节点索引定义
        self.WRIST = 0
        self.THUMB_TIP = 4
        self.INDEX_TIP = 8
        self.MIDDLE_TIP = 12
        self.RING_TIP = 16
        self.PINKY_TIP = 20
        
        self.finger_tips = [4, 8, 12, 16, 20]
        self.finger_mcps = [2, 5, 9, 13, 17]
        self.finger_pips = [3, 7, 11, 15, 19]
        
    def extract_two_hands_landmarks(self, image_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        从图片中提取双手关节点
        
        Args:
            image_path: 图片路径
            
        Returns:
            (左手关节点, 右手关节点)，未检测到的手返回None
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图片: {image_path}")
            return None, None
            
        # 转换为RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 处理图片
        results = self.hands.process(rgb_image)
        
        left_hand_landmarks = None
        right_hand_landmarks = None
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, (hand_landmarks, handedness) in enumerate(zip(
                results.multi_hand_landmarks, results.multi_handedness)):
                
                # 提取关节点
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                
                # 判断左右手
                hand_label = handedness.classification[0].label
                if hand_label == "Left":
                    left_hand_landmarks = landmarks
                else:
                    right_hand_landmarks = landmarks
        
        return left_hand_landmarks, right_hand_landmarks
    
    def normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        标准化单手关节点坐标
        
        Args:
            landmarks: 21x3的关节点数组
            
        Returns:
            标准化后的关节点数组
        """
        normalized = landmarks.copy()
        
        # 1. 平移不变性：以手腕为原点
        wrist = normalized[self.WRIST]
        normalized = normalized - wrist
        
        # 2. 缩放不变性：以手掌长度为单位
        palm_length = np.linalg.norm(normalized[9] - normalized[0])
        if palm_length > 0:
            normalized = normalized / palm_length
        
        return normalized
    
    def normalize_two_hands(self, left_landmarks: Optional[np.ndarray], 
                           right_landmarks: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        标准化双手关节点，考虑双手的相对位置关系
        
        Args:
            left_landmarks: 左手关节点
            right_landmarks: 右手关节点
            
        Returns:
            (标准化左手, 标准化右手)
        """
        # 如果只有一只手，直接标准化
        if left_landmarks is None:
            return None, self.normalize_landmarks(right_landmarks) if right_landmarks is not None else None
        if right_landmarks is None:
            return self.normalize_landmarks(left_landmarks), None
        
        # 双手都存在时，使用统一的参考系
        left_norm = left_landmarks.copy()
        right_norm = right_landmarks.copy()
        
        # 计算双手中心点作为原点
        left_wrist = left_norm[self.WRIST]
        right_wrist = right_norm[self.WRIST]
        center_point = (left_wrist + right_wrist) / 2
        
        # 平移到中心点
        left_norm = left_norm - center_point
        right_norm = right_norm - center_point
        
        # 使用两手腕距离作为缩放参考
        wrist_distance = np.linalg.norm(left_wrist - right_wrist)
        if wrist_distance > 0:
            left_norm = left_norm / wrist_distance
            right_norm = right_norm / wrist_distance
        
        return left_norm, right_norm
    
    def extract_single_hand_features(self, landmarks: np.ndarray) -> np.ndarray:
        """
        提取单手几何特征向量
        
        Args:
            landmarks: 标准化后的关节点
            
        Returns:
            特征向量
        """
        features = []
        
        # 1. 手指长度特征（5个）
        for tip, mcp in zip(self.finger_tips, self.finger_mcps):
            finger_length = np.linalg.norm(landmarks[tip] - landmarks[mcp])
            features.append(finger_length)
        
        # 2. 手指弯曲度特征（5个）
        for tip, pip, mcp in zip(self.finger_tips, self.finger_pips, self.finger_mcps):
            v1 = landmarks[pip] - landmarks[mcp]
            v2 = landmarks[tip] - landmarks[pip]
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            features.append(cos_angle)
        
        # 3. 手指间夹角特征（4个）
        for i in range(len(self.finger_tips) - 1):
            v1 = landmarks[self.finger_tips[i]] - landmarks[self.WRIST]
            v2 = landmarks[self.finger_tips[i + 1]] - landmarks[self.WRIST]
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle)
            features.append(angle)
        
        # 4. 手指相对高度特征（5个）
        palm_plane_y = landmarks[self.WRIST][1]
        for tip in self.finger_tips:
            relative_height = landmarks[tip][1] - palm_plane_y
            features.append(relative_height)
        
        return np.array(features)
    
    def extract_two_hands_interaction_features(self, left_landmarks: Optional[np.ndarray], 
                                              right_landmarks: Optional[np.ndarray]) -> np.ndarray:
        """
        提取双手交互特征
        
        Args:
            left_landmarks: 左手关节点
            right_landmarks: 右手关节点
            
        Returns:
            双手交互特征向量
        """
        features = []
        
        if left_landmarks is not None and right_landmarks is not None:
            # 1. 双手手腕距离
            wrist_distance = np.linalg.norm(left_landmarks[self.WRIST] - right_landmarks[self.WRIST])
            features.append(wrist_distance)
            
            # 2. 双手手掌朝向（使用手掌法向量的夹角）
            left_palm_normal = self.calculate_palm_normal(left_landmarks)
            right_palm_normal = self.calculate_palm_normal(right_landmarks)
            palm_angle = np.arccos(np.clip(np.dot(left_palm_normal, right_palm_normal), -1, 1))
            features.append(palm_angle)
            
            # 3. 对应手指尖距离（5个特征）
            for tip_idx in self.finger_tips:
                tip_distance = np.linalg.norm(left_landmarks[tip_idx] - right_landmarks[tip_idx])
                features.append(tip_distance)
            
            # 4. 双手重心距离
            left_center = np.mean(left_landmarks, axis=0)
            right_center = np.mean(right_landmarks, axis=0)
            center_distance = np.linalg.norm(left_center - right_center)
            features.append(center_distance)
            
            # 5. 双手相对角度（手腕到中指的向量夹角）
            left_direction = left_landmarks[self.MIDDLE_TIP] - left_landmarks[self.WRIST]
            right_direction = right_landmarks[self.MIDDLE_TIP] - right_landmarks[self.WRIST]
            direction_angle = np.arccos(np.clip(
                np.dot(left_direction, right_direction) / 
                (np.linalg.norm(left_direction) * np.linalg.norm(right_direction) + 1e-6), -1, 1))
            features.append(direction_angle)
            
        else:
            # 如果只有一只手，填充零特征
            features = [0] * 10
        
        return np.array(features)
    
    def calculate_palm_normal(self, landmarks: np.ndarray) -> np.ndarray:
        """
        计算手掌法向量
        
        Args:
            landmarks: 手部关节点
            
        Returns:
            归一化的手掌法向量
        """
        # 使用手腕、食指根部、小指根部三点计算平面法向量
        wrist = landmarks[self.WRIST]
        index_mcp = landmarks[5]
        pinky_mcp = landmarks[17]
        
        v1 = index_mcp - wrist
        v2 = pinky_mcp - wrist
        
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm > 0:
            normal = normal / norm
        
        return normal
    
    def extract_combined_features(self, left_landmarks: Optional[np.ndarray], 
                                 right_landmarks: Optional[np.ndarray]) -> np.ndarray:
        """
        提取双手组合特征向量
        
        Args:
            left_landmarks: 左手关节点
            right_landmarks: 右手关节点
            
        Returns:
            组合特征向量
        """
        features = []
        
        # 提取左手特征
        if left_landmarks is not None:
            left_features = self.extract_single_hand_features(left_landmarks)
            features.extend(left_features)
        else:
            features.extend([0] * 19)  # 单手特征维度
        
        # 提取右手特征
        if right_landmarks is not None:
            right_features = self.extract_single_hand_features(right_landmarks)
            features.extend(right_features)
        else:
            features.extend([0] * 19)  # 单手特征维度
        
        # 提取双手交互特征
        interaction_features = self.extract_two_hands_interaction_features(left_landmarks, right_landmarks)
        features.extend(interaction_features)
        
        return np.array(features)
    
    def calculate_two_hands_similarity(self, template: GestureTemplate, 
                                      left_landmarks: Optional[np.ndarray], 
                                      right_landmarks: Optional[np.ndarray]) -> float:
        """
        计算双手手势相似度
        
        Args:
            template: 手势模板
            left_landmarks: 左手关节点
            right_landmarks: 右手关节点
            
        Returns:
            相似度分数 [0, 1]
        """
        # 标准化关节点
        left_norm, right_norm = self.normalize_two_hands(left_landmarks, right_landmarks)
        
        # 提取组合特征
        current_features = self.extract_combined_features(left_norm, right_norm)
        
        # 根据模板类型计算相似度
        if template.hand_type == HandType.BOTH:
            # 双手模板，需要两只手都匹配
            if left_landmarks is None or right_landmarks is None:
                return 0.0  # 缺少一只手，相似度为0
            
            # 计算特征相似度
            feature_distance = np.linalg.norm(current_features - template.combined_features)
            feature_sim = 1 / (1 + feature_distance)
            
            # 计算余弦相似度
            cosine_sim = np.dot(current_features, template.combined_features) / (
                np.linalg.norm(current_features) * np.linalg.norm(template.combined_features) + 1e-6)
            cosine_sim = (cosine_sim + 1) / 2  # 映射到[0, 1]
            
            # 加权组合
            similarity = 0.6 * feature_sim + 0.4 * cosine_sim
            
        elif template.hand_type == HandType.LEFT:
            # 左手模板
            if left_landmarks is None:
                return 0.0
            
            left_features = self.extract_single_hand_features(left_norm)
            feature_distance = np.linalg.norm(left_features - template.left_features)
            similarity = 1 / (1 + feature_distance)
            
        else:  # HandType.RIGHT
            # 右手模板
            if right_landmarks is None:
                return 0.0
            
            right_features = self.extract_single_hand_features(right_norm)
            feature_distance = np.linalg.norm(right_features - template.right_features)
            similarity = 1 / (1 + feature_distance)
        
        return similarity
    
    def add_template(self, name: str, image_path: str, hand_type: HandType = HandType.BOTH) -> bool:
        """
        添加手势模板
        
        Args:
            name: 手势名称
            image_path: 模板图片路径
            hand_type: 手的类型（左手/右手/双手）
            
        Returns:
            是否成功添加
        """
        left_landmarks, right_landmarks = self.extract_two_hands_landmarks(image_path)
        
        # 检查是否提取到所需的手
        if hand_type == HandType.BOTH and (left_landmarks is None or right_landmarks is None):
            print(f"双手模板需要检测到两只手，但图片 {image_path} 中未检测到两只手")
            return False
        elif hand_type == HandType.LEFT and left_landmarks is None:
            print(f"左手模板需要检测到左手，但图片 {image_path} 中未检测到左手")
            return False
        elif hand_type == HandType.RIGHT and right_landmarks is None:
            print(f"右手模板需要检测到右手，但图片 {image_path} 中未检测到右手")
            return False
        
        # 标准化
        left_norm, right_norm = self.normalize_two_hands(left_landmarks, right_landmarks)
        
        # 提取特征
        left_features = self.extract_single_hand_features(left_norm) if left_norm is not None else None
        right_features = self.extract_single_hand_features(right_norm) if right_norm is not None else None
        combined_features = self.extract_combined_features(left_norm, right_norm)
        
        # 创建模板
        template = GestureTemplate(
            name=name,
            hand_type=hand_type,
            left_landmarks=left_norm,
            right_landmarks=right_norm,
            left_features=left_features,
            right_features=right_features,
            combined_features=combined_features,
            image_path=image_path
        )
        
        self.templates[name] = template
        print(f"成功添加{hand_type.value}手势模板: {name}")
        return True
    
    def match_gesture(self, image_path: str, threshold: float = 0.7) -> Tuple[Optional[str], float, Dict[str, float]]:
        """
        匹配手势
        
        Args:
            image_path: 待匹配图片路径
            threshold: 相似度阈值
            
        Returns:
            (最佳匹配手势名, 最高相似度, 所有相似度字典)
        """
        # 提取关节点
        left_landmarks, right_landmarks = self.extract_two_hands_landmarks(image_path)
        
        if left_landmarks is None and right_landmarks is None:
            print("未检测到手部")
            return None, 0.0, {}
        
        # 检测到的手的类型
        detected_hands = []
        if left_landmarks is not None:
            detected_hands.append("左手")
        if right_landmarks is not None:
            detected_hands.append("右手")
        print(f"检测到: {', '.join(detected_hands)}")
        
        # 与所有模板比较
        similarities = {}
        best_match = None
        best_similarity = 0.0
        
        for name, template in self.templates.items():
            similarity = self.calculate_two_hands_similarity(template, left_landmarks, right_landmarks)
            similarities[name] = similarity
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = name
        
        # 检查是否超过阈值
        if best_similarity < threshold:
            best_match = None
        
        return best_match, best_similarity, similarities
    
    def visualize_matching(self, image_path: str, output_path: str = None):
        """
        可视化匹配结果
        
        Args:
            image_path: 待匹配图片路径
            output_path: 输出图片路径（可选）
        """
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图片: {image_path}")
            return
        
        # 获取匹配结果
        best_match, best_similarity, all_similarities = self.match_gesture(image_path)
        
        # 绘制手部关节点
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)
        
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # 绘制关节点和连接
                self.mp_drawing.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # 标注左右手
                hand_label = handedness.classification[0].label
                wrist_pos = hand_landmarks.landmark[self.WRIST]
                x = int(wrist_pos.x * image.shape[1])
                y = int(wrist_pos.y * image.shape[0])
                cv2.putText(image, hand_label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)
        
        # 添加匹配结果文字
        y_offset = 30
        if best_match:
            text = f"Match: {best_match} ({best_similarity:.2%})"
            cv2.putText(image, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            y_offset += 30
        
        # 显示所有相似度
        for name, sim in sorted(all_similarities.items(), key=lambda x: x[1], reverse=True):
            template_type = self.templates[name].hand_type.value if name in self.templates else ""
            text = f"{name} ({template_type}): {sim:.2%}"
            color = (0, 255, 0) if sim > 0.7 else (0, 165, 255) if sim > 0.5 else (0, 0, 255)
            cv2.putText(image, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            y_offset += 25
        
        # 显示或保存结果
        if output_path:
            cv2.imwrite(output_path, image)
            print(f"结果已保存到: {output_path}")
        else:
            cv2.imshow("Two Hands Gesture Matching", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    def save_templates(self, filepath: str = "two_hands_templates.json"):
        """保存模板到文件"""
        data = {}
        for name, template in self.templates.items():
            data[name] = {
                'hand_type': template.hand_type.value,
                'left_landmarks': template.left_landmarks.tolist() if template.left_landmarks is not None else None,
                'right_landmarks': template.right_landmarks.tolist() if template.right_landmarks is not None else None,
                'left_features': template.left_features.tolist() if template.left_features is not None else None,
                'right_features': template.right_features.tolist() if template.right_features is not None else None,
                'combined_features': template.combined_features.tolist(),
                'image_path': template.image_path
            }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"模板已保存到: {filepath}")
    
    def load_templates(self, filepath: str = "two_hands_templates.json"):
        """从文件加载模板"""
        if not os.path.exists(filepath):
            print(f"模板文件不存在: {filepath}")
            return
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        for name, template_data in data.items():
            template = GestureTemplate(
                name=name,
                hand_type=HandType(template_data['hand_type']),
                left_landmarks=np.array(template_data['left_landmarks']) if template_data['left_landmarks'] else None,
                right_landmarks=np.array(template_data['right_landmarks']) if template_data['right_landmarks'] else None,
                left_features=np.array(template_data['left_features']) if template_data['left_features'] else None,
                right_features=np.array(template_data['right_features']) if template_data['right_features'] else None,
                combined_features=np.array(template_data['combined_features']),
                image_path=template_data['image_path']
            )
            self.templates[name] = template
        
        print(f"已加载 {len(self.templates)} 个模板")


def main():
    """主函数：演示双手手势识别系统的使用"""
    
    # 创建识别器
    recognizer = TwoHandGestureRecognizer()
    
    # 1. 添加模板（需要准备模板图片）
    print("=== 添加双手手势模板 ===")
    
    # 双手手势模板
    two_hands_templates = {
        "heart": ("templates/heart.jpg", HandType.BOTH),           # 双手比心
        "prayer": ("templates/prayer.jpg", HandType.BOTH),         # 双手合十
        "cross": ("templates/cross.jpg", HandType.BOTH),           # 双手交叉
        "butterfly": ("templates/butterfly.jpg", HandType.BOTH),   # 双手蝴蝶
        "frame": ("templates/frame.jpg", HandType.BOTH),           # 双手框架
    }
    
    # 单手手势模板（可选）
    single_hand_templates = {
        "thumbs_up_left": ("templates/thumbs_up_left.jpg", HandType.LEFT),
        "peace_right": ("templates/peace_right.jpg", HandType.RIGHT),
    }
    
    # 检查并创建模板目录
    os.makedirs("templates", exist_ok=True)
    
    # 添加双手模板
    for gesture_name, (image_path, hand_type) in two_hands_templates.items():
        if os.path.exists(image_path):
            recognizer.add_template(gesture_name, image_path, hand_type)
        else:
            print(f"模板图片不存在: {image_path}")
    
    # 添加单手模板
    for gesture_name, (image_path, hand_type) in single_hand_templates.items():
        if os.path.exists(image_path):
            recognizer.add_template(gesture_name, image_path, hand_type)
        else:
            print(f"模板图片不存在: {image_path}")
    
    # 2. 保存模板（可选）
    if recognizer.templates:
        recognizer.save_templates("two_hands_gesture_templates.json")
    
    # 3. 测试识别
    print("\n=== 测试双手手势识别 ===")
    test_image = "test_two_hands.jpg"
    
    if os.path.exists(test_image):
        # 执行匹配
        match, similarity, all_similarities = recognizer.match_gesture(test_image)
        
        print(f"\n测试图片: {test_image}")
        if match:
            print(f"最佳匹配: {match} (相似度: {similarity:.2%})")
        else:
            print("未找到匹配的手势")
        
        print("\n所有手势相似度:")
        for name, sim in sorted(all_similarities.items(), key=lambda x: x[1], reverse=True):
            template_type = recognizer.templates[name].hand_type.value if name in recognizer.templates else ""
            print(f"  {name} ({template_type}): {sim:.2%}")
        
        # 可视化结果
        recognizer.visualize_matching(test_image, "two_hands_result.jpg")
    else:
        print(f"测试图片不存在: {test_image}")
        print("\n使用方法:")
        print("1. 准备双手手势模板图片放在 templates/ 目录下")
        print("2. 准备测试图片命名为 test_two_hands.jpg")
        print("3. 运行程序进行匹配")
        
    # 4. 批量测试（可选）
    print("\n=== 批量测试模式 ===")
    test_dir = "test_images"
    if os.path.exists(test_dir):
        for test_file in os.listdir(test_dir):
            if test_file.endswith(('.jpg', '.png', '.jpeg')):
                test_path = os.path.join(test_dir, test_file)
                print(f"\n测试: {test_file}")
                match, similarity, _ = recognizer.match_gesture(test_path)
                if match:
                    print(f"  匹配: {match} ({similarity:.2%})")
                else:
                    print("  未匹配")


if __name__ == "__main__":
    main()