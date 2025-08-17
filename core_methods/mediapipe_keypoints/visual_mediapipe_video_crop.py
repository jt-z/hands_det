#!/usr/bin/env python3
"""
MediaPipe手部检测视频处理脚本
支持处理MP4视频文件，检测最多两只手，可视化关键点并保存结果
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import os
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Tuple
from enum import Enum
from datetime import datetime


class HandType(Enum):
    """手的类型"""
    LEFT = "left"
    RIGHT = "right"
    UNKNOWN = "unknown"


@dataclass
class HandLandmarks:
    """单手关键点数据结构"""
    hand_type: HandType
    landmarks: np.ndarray  # 21个关键点，每个点(x, y, z)
    confidence: float
    timestamp: float


@dataclass
class FrameResult:
    """单帧检测结果"""
    frame_number: int
    timestamp: float
    left_hand: Optional[HandLandmarks]
    right_hand: Optional[HandLandmarks]
    image_width: int
    image_height: int


class MediaPipeHandDetector:
    """MediaPipe手部检测器"""
    
    def __init__(self):
        """初始化MediaPipe手部检测器"""
        # MediaPipe初始化
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # 配置检测器
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,  # 视频模式
            max_num_hands=2,  # 最多检测两只手
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 关键点索引定义
        self.WRIST = 0
        self.THUMB_TIP = 4
        self.INDEX_TIP = 8
        self.MIDDLE_TIP = 12
        self.RING_TIP = 16
        self.PINKY_TIP = 20
        
        self.finger_tips = [4, 8, 12, 16, 20]
        self.finger_mcps = [2, 5, 9, 13, 17]  # 掌指关节点
        self.finger_pips = [3, 7, 11, 15, 19]
    
    def detect_hands_in_frame(self, image: np.ndarray, frame_number: int, timestamp: float) -> FrameResult:
        """
        在单帧图像中检测手部
        
        Args:
            image: 输入图像 (BGR格式)
            frame_number: 帧号
            timestamp: 时间戳
            
        Returns:
            FrameResult: 检测结果
        """
        # 转换为RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 处理图像
        results = self.hands.process(rgb_image)
        
        left_hand = None
        right_hand = None
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # 确定手的类型
                hand_label = handedness.classification[0].label
                hand_confidence = handedness.classification[0].score
                
                # 注意：MediaPipe的left/right是基于图像视角的，需要转换
                if hand_label == "Left":
                    hand_type = HandType.RIGHT  # 图像中的左手实际是人的右手
                else:
                    hand_type = HandType.LEFT   # 图像中的右手实际是人的左手
                
                # 提取关键点坐标
                landmarks_array = np.array([
                    [landmark.x, landmark.y, landmark.z] 
                    for landmark in hand_landmarks.landmark
                ])
                
                # 创建HandLandmarks对象
                hand_data = HandLandmarks(
                    hand_type=hand_type,
                    landmarks=landmarks_array,
                    confidence=hand_confidence,
                    timestamp=timestamp
                )
                
                # 分配给对应的手
                if hand_type == HandType.LEFT:
                    left_hand = hand_data
                elif hand_type == HandType.RIGHT:
                    right_hand = hand_data
        
        return FrameResult(
            frame_number=frame_number,
            timestamp=timestamp,
            left_hand=left_hand,
            right_hand=right_hand,
            image_width=image.shape[1],
            image_height=image.shape[0]
        )

    def calculate_hand_direction_and_crop_box(self, landmarks: np.ndarray, image_width: int, image_height: int, 
                                             padding: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        根据手腕和掌指关节点计算手臂方向，确定裁剪区域
        
        Args:
            landmarks: 手部关键点数组 (21, 3)
            image_width: 图像宽度
            image_height: 图像高度
            padding: 裁剪时的边界填充
            
        Returns:
            crop_coords: 裁剪坐标 [x1, y1, x2, y2]
            arm_direction: 手臂方向向量 [dx, dy]
        """
        # 获取关键点的像素坐标
        wrist = landmarks[self.WRIST][:2] * np.array([image_width, image_height])
        
        # 获取掌指关节点 (MCP points: 2, 5, 9, 13, 17)
        mcp_points = []
        for mcp_idx in self.finger_mcps:
            mcp_point = landmarks[mcp_idx][:2] * np.array([image_width, image_height])
            mcp_points.append(mcp_point)
        
        mcp_points = np.array(mcp_points)
        
        # 计算掌指关节点的中心
        mcp_center = np.mean(mcp_points, axis=0)
        
        # 计算从手腕到掌指关节中心的向量（手掌方向）
        hand_direction = mcp_center - wrist
        hand_direction_norm = hand_direction / (np.linalg.norm(hand_direction) + 1e-8)
        
        # 手臂方向是手掌方向的反方向
        arm_direction = -hand_direction_norm
        
        # 计算裁剪位置：从手腕位置向手臂方向延伸一小段距离作为裁剪边界
        crop_offset_distance = 30  # 可调节的偏移距离
        crop_line_point = wrist + arm_direction * crop_offset_distance
        
        # 计算垂直于手臂方向的向量
        perpendicular = np.array([-arm_direction[1], arm_direction[0]])
        
        # 获取所有手部关键点的边界框
        all_points = landmarks[:, :2] * np.array([image_width, image_height])
        
        # 过滤掉手腕以下的点（在手臂方向上）
        valid_points = []
        for point in all_points:
            # 计算点到裁剪线的距离（在手臂方向上的投影）
            to_point = point - crop_line_point
            projection = np.dot(to_point, arm_direction)
            
            # 如果投影为负，说明点在手掌侧，保留
            if projection <= 0:
                valid_points.append(point)
        
        if not valid_points:
            # 如果没有有效点，使用所有点
            valid_points = all_points
        
        valid_points = np.array(valid_points)
        
        # 计算裁剪边界框
        min_x = max(0, int(np.min(valid_points[:, 0])) - padding)
        max_x = min(image_width, int(np.max(valid_points[:, 0])) + padding)
        min_y = max(0, int(np.min(valid_points[:, 1])) - padding)
        max_y = min(image_height, int(np.max(valid_points[:, 1])) + padding)
        
        # 确保裁剪区域是正方形（可选）
        width = max_x - min_x
        height = max_y - min_y
        size = max(width, height)
        
        # 重新计算中心对齐的正方形区域
        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2
        
        half_size = size // 2
        crop_coords = np.array([
            max(0, center_x - half_size),  # x1
            max(0, center_y - half_size),  # y1
            min(image_width, center_x + half_size),   # x2
            min(image_height, center_y + half_size)   # y2
        ])
        
        return crop_coords, arm_direction
    
    def crop_hands_from_frame(self, image: np.ndarray, frame_result: FrameResult, 
                             save_crops: bool = True, output_dir: str = None) -> List[Dict]:
        """
        从帧中裁剪手部区域
        
        Args:
            image: 输入图像
            frame_result: 检测结果
            save_crops: 是否保存裁剪的图像
            output_dir: 输出目录
            
        Returns:
            裁剪信息列表
        """
        crops_info = []
        
        if frame_result.left_hand:
            crop_coords, arm_direction = self.calculate_hand_direction_and_crop_box(
                frame_result.left_hand.landmarks,
                frame_result.image_width,
                frame_result.image_height
            )
            
            # 裁剪图像
            x1, y1, x2, y2 = crop_coords
            cropped_img = image[y1:y2, x1:x2]
            
            crop_info = {
                "hand_type": "left",
                "crop_coords": crop_coords.tolist(),
                "arm_direction": arm_direction.tolist(),
                "cropped_image": cropped_img,
                "original_landmarks": frame_result.left_hand.landmarks
            }
            
            # 保存裁剪图像
            if save_crops and output_dir and cropped_img.size > 0:
                crop_filename = f"frame_{frame_result.frame_number:06d}_left_hand.jpg"
                crop_path = os.path.join(output_dir, "hand_crops", crop_filename)
                os.makedirs(os.path.dirname(crop_path), exist_ok=True)
                cv2.imwrite(crop_path, cropped_img)
                crop_info["saved_path"] = crop_path
            
            crops_info.append(crop_info)
        
        if frame_result.right_hand:
            crop_coords, arm_direction = self.calculate_hand_direction_and_crop_box(
                frame_result.right_hand.landmarks,
                frame_result.image_width,
                frame_result.image_height
            )
            
            # 裁剪图像
            x1, y1, x2, y2 = crop_coords
            cropped_img = image[y1:y2, x1:x2]
            
            crop_info = {
                "hand_type": "right",
                "crop_coords": crop_coords.tolist(),
                "arm_direction": arm_direction.tolist(),
                "cropped_image": cropped_img,
                "original_landmarks": frame_result.right_hand.landmarks
            }
            
            # 保存裁剪图像
            if save_crops and output_dir and cropped_img.size > 0:
                crop_filename = f"frame_{frame_result.frame_number:06d}_right_hand.jpg"
                crop_path = os.path.join(output_dir, "hand_crops", crop_filename)
                os.makedirs(os.path.dirname(crop_path), exist_ok=True)
                cv2.imwrite(crop_path, cropped_img)
                crop_info["saved_path"] = crop_path
            
            crops_info.append(crop_info)
        
        return crops_info
        """
        在单帧图像中检测手部
        
        Args:
            image: 输入图像 (BGR格式)
            frame_number: 帧号
            timestamp: 时间戳
            
        Returns:
            FrameResult: 检测结果
        """
        # 转换为RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 处理图像
        results = self.hands.process(rgb_image)
        
        left_hand = None
        right_hand = None
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # 确定手的类型
                hand_label = handedness.classification[0].label
                hand_confidence = handedness.classification[0].score
                
                # 注意：MediaPipe的left/right是基于图像视角的，需要转换
                if hand_label == "Left":
                    hand_type = HandType.RIGHT  # 图像中的左手实际是人的右手
                else:
                    hand_type = HandType.LEFT   # 图像中的右手实际是人的左手
                
                # 提取关键点坐标
                landmarks_array = np.array([
                    [landmark.x, landmark.y, landmark.z] 
                    for landmark in hand_landmarks.landmark
                ])
                
                # 创建HandLandmarks对象
                hand_data = HandLandmarks(
                    hand_type=hand_type,
                    landmarks=landmarks_array,
                    confidence=hand_confidence,
                    timestamp=timestamp
                )
                
                # 分配给对应的手
                if hand_type == HandType.LEFT:
                    left_hand = hand_data
                elif hand_type == HandType.RIGHT:
                    right_hand = hand_data
        
        return FrameResult(
            frame_number=frame_number,
            timestamp=timestamp,
            left_hand=left_hand,
            right_hand=right_hand,
            image_width=image.shape[1],
            image_height=image.shape[0]
        )
    
    def draw_landmarks_on_image(self, image: np.ndarray, frame_result: FrameResult, 
                               show_crop_boxes: bool = True) -> np.ndarray:
        """
        在图像上绘制手部关键点和裁剪框
        
        Args:
            image: 输入图像
            frame_result: 检测结果
            show_crop_boxes: 是否显示裁剪框
            
        Returns:
            绘制了关键点的图像
        """
        annotated_image = image.copy()
        
        # 绘制左手
        if frame_result.left_hand:
            self._draw_single_hand(
                annotated_image, 
                frame_result.left_hand, 
                color=(0, 255, 0),  # 绿色
                label="Left Hand"
            )
            
            # 绘制裁剪框
            if show_crop_boxes:
                crop_coords, arm_direction = self.calculate_hand_direction_and_crop_box(
                    frame_result.left_hand.landmarks,
                    frame_result.image_width,
                    frame_result.image_height
                )
                self._draw_crop_box(annotated_image, crop_coords, arm_direction, (0, 255, 0))
        
        # 绘制右手
        if frame_result.right_hand:
            self._draw_single_hand(
                annotated_image, 
                frame_result.right_hand, 
                color=(255, 0, 0),  # 蓝色
                label="Right Hand"
            )
            
            # 绘制裁剪框
            if show_crop_boxes:
                crop_coords, arm_direction = self.calculate_hand_direction_and_crop_box(
                    frame_result.right_hand.landmarks,
                    frame_result.image_width,
                    frame_result.image_height
                )
                self._draw_crop_box(annotated_image, crop_coords, arm_direction, (255, 0, 0))
        
        # 添加帧信息
        cv2.putText(
            annotated_image,
            f"Frame: {frame_result.frame_number}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )
        
        return annotated_image
    
    def _draw_crop_box(self, image: np.ndarray, crop_coords: np.ndarray, 
                      arm_direction: np.ndarray, color: Tuple[int, int, int]):
        """绘制裁剪框和手臂方向"""
        x1, y1, x2, y2 = crop_coords.astype(int)
        
        # 绘制裁剪框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # 绘制手臂方向箭头
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        arrow_length = 50
        end_x = int(center_x + arm_direction[0] * arrow_length)
        end_y = int(center_y + arm_direction[1] * arrow_length)
        
        cv2.arrowedLine(image, (center_x, center_y), (end_x, end_y), color, 2, tipLength=0.3)
        
        # 添加标签
        cv2.putText(
            image,
            "Crop Box",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )
    
    def _draw_single_hand(self, image: np.ndarray, hand_data: HandLandmarks, color: Tuple[int, int, int], label: str):
        """绘制单手关键点"""
        h, w = image.shape[:2]
        
        # 绘制关键点
        for i, (x, y, z) in enumerate(hand_data.landmarks):
            # 转换为像素坐标
            px, py = int(x * w), int(y * h)
            
            # 绘制关键点
            cv2.circle(image, (px, py), 5, color, -1)
            
            # 绘制关键点编号
            cv2.putText(
                image,
                str(i),
                (px + 5, py - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                color,
                1
            )
        
        # 绘制连接线
        self._draw_hand_connections(image, hand_data.landmarks, w, h, color)
        
        # 添加手的标签和置信度
        if len(hand_data.landmarks) > 0:
            wrist_x, wrist_y = int(hand_data.landmarks[0][0] * w), int(hand_data.landmarks[0][1] * h)
            cv2.putText(
                image,
                f"{label} ({hand_data.confidence:.2f})",
                (wrist_x - 50, wrist_y - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
    
    def _draw_hand_connections(self, image: np.ndarray, landmarks: np.ndarray, w: int, h: int, color: Tuple[int, int, int]):
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
                cv2.line(image, start_point, end_point, color, 2)


class VideoProcessor:
    """视频处理器"""
    
    def __init__(self, output_dir: str = "output"):
        """
        初始化视频处理器
        
        Args:
            output_dir: 输出目录
        """
        self.detector = MediaPipeHandDetector()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 结果存储
        self.frame_results: List[FrameResult] = []
        self.crops_info: List[List[Dict]] = []  # 每帧的裁剪信息
    
    def process_video(self, video_path: str, output_video_path: Optional[str] = None, 
                     save_json: bool = True, save_hand_crops: bool = True, 
                     frame_skip: int = 1, show_crop_boxes: bool = True) -> bool:
        """
        处理视频文件
        
        Args:
            video_path: 输入视频路径
            output_video_path: 输出视频路径（可选）
            save_json: 是否保存JSON结果
            save_hand_crops: 是否保存手部裁剪图像
            frame_skip: 帧跳过间隔（1表示处理每一帧）
            show_crop_boxes: 是否在输出视频中显示裁剪框
            
        Returns:
            处理是否成功
        """
        if not os.path.exists(video_path):
            print(f"错误：视频文件不存在: {video_path}")
            return False
        
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"错误：无法打开视频文件: {video_path}")
            return False
        
        # 获取视频信息
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"视频信息: {frame_width}x{frame_height}, {fps}fps, 总帧数: {total_frames}")
        
        # 设置输出视频
        video_writer = None
        if output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        
        # 清空之前的结果
        self.frame_results = []
        self.crops_info = []
        
        frame_number = 0
        processed_frames = 0
        
        print("开始处理视频...")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 跳帧处理
                if frame_number % frame_skip != 0:
                    frame_number += 1
                    continue
                
                # 计算时间戳
                timestamp = frame_number / fps
                
                # 检测手部
                frame_result = self.detector.detect_hands_in_frame(frame, frame_number, timestamp)
                self.frame_results.append(frame_result)
                
                # 裁剪手部区域
                crops_info = []
                if save_hand_crops and (frame_result.left_hand or frame_result.right_hand):
                    crops_info = self.detector.crop_hands_from_frame(
                        frame, frame_result, save_crops=True, output_dir=str(self.output_dir)
                    )
                self.crops_info.append(crops_info)
                
                # 绘制关键点和裁剪框
                annotated_frame = self.detector.draw_landmarks_on_image(
                    frame, frame_result, show_crop_boxes=show_crop_boxes
                )
                
                # 写入输出视频
                if video_writer:
                    video_writer.write(annotated_frame)
                
                processed_frames += 1
                
                # 显示进度
                if processed_frames % 30 == 0:
                    progress = (frame_number / total_frames) * 100
                    hands_detected = len([c for c in crops_info if c])
                    print(f"处理进度: {progress:.1f}% ({frame_number}/{total_frames}), 当前帧检测到 {hands_detected} 只手")
                
                frame_number += 1
            
            print(f"视频处理完成，共处理 {processed_frames} 帧")
            
            # 保存JSON结果
            if save_json:
                self._save_results_to_json(video_path)
            
            # 统计裁剪信息
            if save_hand_crops:
                self._print_crop_statistics()
            
            return True
            
        except Exception as e:
            print(f"处理视频时出错: {e}")
            return False
            
        finally:
            # 释放资源
            cap.release()
            if video_writer:
                video_writer.release()
    
    def _print_crop_statistics(self):
        """打印裁剪统计信息"""
        total_left_crops = sum(1 for crops in self.crops_info for crop in crops if crop.get("hand_type") == "left")
        total_right_crops = sum(1 for crops in self.crops_info for crops in crops if crop.get("hand_type") == "right")
        
        print(f"\n手部裁剪统计:")
        print(f"  左手裁剪图像: {total_left_crops} 张")
        print(f"  右手裁剪图像: {total_right_crops} 张")
        print(f"  总裁剪图像: {total_left_crops + total_right_crops} 张")
        print(f"  保存路径: {self.output_dir}/hand_crops/")
    
    def _save_results_to_json(self, video_path: str):
        """保存检测结果到JSON文件"""
        # 准备JSON数据
        json_data = {
            "video_info": {
                "source_path": video_path,
                "processed_time": datetime.now().isoformat(),
                "total_frames": len(self.frame_results)
            },
            "detection_results": []
        }
        
        for i, frame_result in enumerate(self.frame_results):
            frame_data = {
                "frame_number": frame_result.frame_number,
                "timestamp": frame_result.timestamp,
                "image_width": frame_result.image_width,
                "image_height": frame_result.image_height,
                "hands": {},
                "crops": []
            }
            
            # 左手数据
            if frame_result.left_hand:
                frame_data["hands"]["left"] = {
                    "confidence": frame_result.left_hand.confidence,
                    "landmarks": frame_result.left_hand.landmarks.tolist()
                }
            
            # 右手数据
            if frame_result.right_hand:
                frame_data["hands"]["right"] = {
                    "confidence": frame_result.right_hand.confidence,
                    "landmarks": frame_result.right_hand.landmarks.tolist()
                }
            
            # 裁剪信息
            if i < len(self.crops_info):
                for crop_info in self.crops_info[i]:
                    crop_data = {
                        "hand_type": crop_info["hand_type"],
                        "crop_coords": crop_info["crop_coords"],
                        "arm_direction": crop_info["arm_direction"]
                    }
                    if "saved_path" in crop_info:
                        crop_data["saved_path"] = crop_info["saved_path"]
                    frame_data["crops"].append(crop_data)
            
            json_data["detection_results"].append(frame_data)
        
        # 保存JSON文件
        video_name = Path(video_path).stem
        json_path = self.output_dir / f"{video_name}_hand_detection_with_crops.json"
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"检测结果已保存到: {json_path}")
        for frame_result in self.frame_results:
            frame_data = {
                "frame_number": frame_result.frame_number,
                "timestamp": frame_result.timestamp,
                "image_width": frame_result.image_width,
                "image_height": frame_result.image_height,
                "hands": {}
            }
            
            # 左手数据
            if frame_result.left_hand:
                frame_data["hands"]["left"] = {
                    "confidence": frame_result.left_hand.confidence,
                    "landmarks": frame_result.left_hand.landmarks.tolist()
                }
            
            # 右手数据
            if frame_result.right_hand:
                frame_data["hands"]["right"] = {
                    "confidence": frame_result.right_hand.confidence,
                    "landmarks": frame_result.right_hand.landmarks.tolist()
                }
            
            json_data["detection_results"].append(frame_data)
        
        # 保存JSON文件
        video_name = Path(video_path).stem
        json_path = self.output_dir / f"{video_name}_hand_detection.json"
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"检测结果已保存到: {json_path}")
    
    def get_statistics(self) -> Dict:
        """获取检测统计信息"""
        if not self.frame_results:
            return {}
        
        total_frames = len(self.frame_results)
        frames_with_left_hand = sum(1 for r in self.frame_results if r.left_hand)
        frames_with_right_hand = sum(1 for r in self.frame_results if r.right_hand)
        frames_with_both_hands = sum(1 for r in self.frame_results if r.left_hand and r.right_hand)
        frames_with_any_hand = sum(1 for r in self.frame_results if r.left_hand or r.right_hand)
        
        return {
            "total_frames": total_frames,
            "frames_with_left_hand": frames_with_left_hand,
            "frames_with_right_hand": frames_with_right_hand,
            "frames_with_both_hands": frames_with_both_hands,
            "frames_with_any_hand": frames_with_any_hand,
            "left_hand_detection_rate": frames_with_left_hand / total_frames * 100,
            "right_hand_detection_rate": frames_with_right_hand / total_frames * 100,
            "both_hands_detection_rate": frames_with_both_hands / total_frames * 100,
            "any_hand_detection_rate": frames_with_any_hand / total_frames * 100
        }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='MediaPipe手部检测视频处理工具')
    parser.add_argument('video_path', help='输入视频文件路径（MP4格式）')
    parser.add_argument('--output-dir', default='output', help='输出目录')
    parser.add_argument('--output-video', help='输出视频文件路径（可选）')
    parser.add_argument('--no-json', action='store_true', help='不保存JSON结果')
    parser.add_argument('--no-crops', action='store_true', help='不保存手部裁剪图像')
    parser.add_argument('--no-crop-boxes', action='store_true', help='不在视频中显示裁剪框')
    parser.add_argument('--frame-skip', type=int, default=1, help='帧跳过间隔（默认处理每一帧）')
    parser.add_argument('--crop-padding', type=int, default=50, help='裁剪时的边界填充像素（默认50）')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.video_path):
        print(f"错误：输入视频文件不存在: {args.video_path}")
        return
    
    # 生成默认输出视频路径
    if not args.output_video:
        video_name = Path(args.video_path).stem
        args.output_video = os.path.join(args.output_dir, f"{video_name}_with_landmarks_and_crops.mp4")
    
    # 创建视频处理器
    processor = VideoProcessor(args.output_dir)
    
    # 处理视频
    success = processor.process_video(
        video_path=args.video_path,
        output_video_path=args.output_video,
        save_json=not args.no_json,
        save_hand_crops=not args.no_crops,
        frame_skip=args.frame_skip,
        show_crop_boxes=not args.no_crop_boxes
    )
    
    if success:
        print(f"处理成功！输出文件:")
        print(f"  视频: {args.output_video}")
        if not args.no_json:
            video_name = Path(args.video_path).stem
            json_path = os.path.join(args.output_dir, f"{video_name}_hand_detection_with_crops.json")
            print(f"  JSON: {json_path}")
        if not args.no_crops:
            print(f"  手部裁剪图像: {args.output_dir}/hand_crops/")
        
        # 显示统计信息
        stats = processor.get_statistics()
        print(f"\n检测统计:")
        print(f"  总帧数: {stats['total_frames']}")
        print(f"  检测到左手的帧数: {stats['frames_with_left_hand']} ({stats['left_hand_detection_rate']:.1f}%)")
        print(f"  检测到右手的帧数: {stats['frames_with_right_hand']} ({stats['right_hand_detection_rate']:.1f}%)")
        print(f"  检测到双手的帧数: {stats['frames_with_both_hands']} ({stats['both_hands_detection_rate']:.1f}%)")
        print(f"  检测到任意手的帧数: {stats['frames_with_any_hand']} ({stats['any_hand_detection_rate']:.1f}%)")
    else:
        print("处理失败！")


if __name__ == "__main__":
    main()