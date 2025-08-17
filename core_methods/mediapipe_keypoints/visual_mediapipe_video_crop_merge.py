#!/usr/bin/env python3
"""
MediaPipe手部检测视频处理脚本
支持处理MP4视频文件，检测最多两只手，可视化关键点并保存结果
改进版：当检测到两只手时，合并裁剪框以包含两只手的完整区域
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

    def calculate_single_hand_crop_box(self, landmarks: np.ndarray, image_width: int, image_height: int, 
                                      padding: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算单手的裁剪区域
        
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
        
        crop_coords = np.array([min_x, min_y, max_x, max_y])
        
        return crop_coords, arm_direction

    def calculate_merged_crop_box(self, frame_result: FrameResult, padding: int = 50, 
                                 square_crop: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        计算合并的裁剪框（当有两只手时合并，单手时使用单手框）
        
        Args:
            frame_result: 检测结果
            padding: 裁剪时的边界填充
            square_crop: 是否强制生成正方形裁剪区域
            
        Returns:
            merged_crop_coords: 合并后的裁剪坐标 [x1, y1, x2, y2]
            crop_info: 裁剪信息字典
        """
        crop_info = {
            "has_left_hand": frame_result.left_hand is not None,
            "has_right_hand": frame_result.right_hand is not None,
            "crop_strategy": "none"
        }
        
        # 如果没有检测到手，返回空
        if not frame_result.left_hand and not frame_result.right_hand:
            return np.array([0, 0, 0, 0]), crop_info
        
        # 收集所有有效的手部关键点
        all_valid_points = []
        individual_crops = []
        
        if frame_result.left_hand:
            left_crop, left_arm_dir = self.calculate_single_hand_crop_box(
                frame_result.left_hand.landmarks,
                frame_result.image_width,
                frame_result.image_height,
                padding=0  # 暂时不加padding，后面统一处理
            )
            individual_crops.append(("left", left_crop, left_arm_dir))
            
            # 添加左手的所有关键点到合并点集
            left_points = frame_result.left_hand.landmarks[:, :2] * np.array([
                frame_result.image_width, frame_result.image_height
            ])
            all_valid_points.extend(left_points)
            crop_info["left_crop"] = left_crop.tolist()
            crop_info["left_arm_direction"] = left_arm_dir.tolist()
        
        if frame_result.right_hand:
            right_crop, right_arm_dir = self.calculate_single_hand_crop_box(
                frame_result.right_hand.landmarks,
                frame_result.image_width,
                frame_result.image_height,
                padding=0  # 暂时不加padding，后面统一处理
            )
            individual_crops.append(("right", right_crop, right_arm_dir))
            
            # 添加右手的所有关键点到合并点集
            right_points = frame_result.right_hand.landmarks[:, :2] * np.array([
                frame_result.image_width, frame_result.image_height
            ])
            all_valid_points.extend(right_points)
            crop_info["right_crop"] = right_crop.tolist()
            crop_info["right_arm_direction"] = right_arm_dir.tolist()
        
        # 确定裁剪策略
        if len(individual_crops) == 1:
            # 单手情况
            crop_info["crop_strategy"] = "single_hand"
            hand_type, single_crop, arm_dir = individual_crops[0]
            
            # 添加padding
            merged_crop = np.array([
                max(0, single_crop[0] - padding),
                max(0, single_crop[1] - padding),
                min(frame_result.image_width, single_crop[2] + padding),
                min(frame_result.image_height, single_crop[3] + padding)
            ])
            
        else:
            # 双手情况 - 合并裁剪框
            crop_info["crop_strategy"] = "merged_hands"
            
            # 计算所有关键点的边界框
            all_valid_points = np.array(all_valid_points)
            min_x = max(0, int(np.min(all_valid_points[:, 0])) - padding)
            max_x = min(frame_result.image_width, int(np.max(all_valid_points[:, 0])) + padding)
            min_y = max(0, int(np.min(all_valid_points[:, 1])) - padding)
            max_y = min(frame_result.image_height, int(np.max(all_valid_points[:, 1])) + padding)
            
            merged_crop = np.array([min_x, min_y, max_x, max_y])
        
        # 如果需要正方形裁剪
        if square_crop:
            merged_crop = self._make_square_crop(merged_crop, frame_result.image_width, frame_result.image_height)
            crop_info["is_square"] = True
        else:
            crop_info["is_square"] = False
        
        crop_info["final_crop"] = merged_crop.tolist()
        
        return merged_crop, crop_info

    def _make_square_crop(self, crop_coords: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
        """
        将矩形裁剪区域调整为正方形
        
        Args:
            crop_coords: 原始裁剪坐标 [x1, y1, x2, y2]
            image_width: 图像宽度
            image_height: 图像高度
            
        Returns:
            调整后的正方形裁剪坐标
        """
        x1, y1, x2, y2 = crop_coords
        
        # 计算当前宽度和高度
        width = x2 - x1
        height = y2 - y1
        
        # 取较大的尺寸作为正方形边长
        size = max(width, height)
        
        # 计算中心点
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # 计算新的正方形边界
        half_size = size // 2
        new_x1 = max(0, center_x - half_size)
        new_y1 = max(0, center_y - half_size)
        new_x2 = min(image_width, center_x + half_size)
        new_y2 = min(image_height, center_y + half_size)
        
        # 如果边界被图像边缘限制，需要调整另一边以保持正方形
        actual_width = new_x2 - new_x1
        actual_height = new_y2 - new_y1
        actual_size = min(actual_width, actual_height)
        
        # 重新计算以确保是正方形
        new_x1 = max(0, center_x - actual_size // 2)
        new_y1 = max(0, center_y - actual_size // 2)
        new_x2 = min(image_width, new_x1 + actual_size)
        new_y2 = min(image_height, new_y1 + actual_size)
        
        return np.array([new_x1, new_y1, new_x2, new_y2])
    
    def crop_hands_from_frame(self, image: np.ndarray, frame_result: FrameResult, 
                             save_crops: bool = True, output_dir: str = None,
                             square_crop: bool = True, padding: int = 50) -> Dict:
        """
        从帧中裁剪手部区域（智能合并双手）
        
        Args:
            image: 输入图像
            frame_result: 检测结果
            save_crops: 是否保存裁剪的图像
            output_dir: 输出目录
            square_crop: 是否生成正方形裁剪
            padding: 边界填充
            
        Returns:
            裁剪信息字典
        """
        # 计算合并的裁剪框
        merged_crop_coords, crop_info = self.calculate_merged_crop_box(
            frame_result, padding=padding, square_crop=square_crop
        )
        
        crop_info["frame_number"] = frame_result.frame_number
        crop_info["timestamp"] = frame_result.timestamp
        
        # 如果没有检测到手，返回空信息
        if crop_info["crop_strategy"] == "none":
            return crop_info
        
        # 裁剪图像
        x1, y1, x2, y2 = merged_crop_coords.astype(int)
        
        # 确保裁剪坐标有效
        if x2 > x1 and y2 > y1:
            cropped_img = image[y1:y2, x1:x2]
            crop_info["cropped_image"] = cropped_img
            crop_info["crop_size"] = [x2 - x1, y2 - y1]
            
            # 保存裁剪图像
            if save_crops and output_dir and cropped_img.size > 0:
                strategy_suffix = "merged" if crop_info["crop_strategy"] == "merged_hands" else "single"
                crop_filename = f"frame_{frame_result.frame_number:06d}_{strategy_suffix}_hands.jpg"
                crop_path = os.path.join(output_dir, "hand_crops", crop_filename)
                os.makedirs(os.path.dirname(crop_path), exist_ok=True)
                cv2.imwrite(crop_path, cropped_img)
                crop_info["saved_path"] = crop_path
        else:
            crop_info["error"] = "Invalid crop coordinates"
        
        return crop_info
    
    def draw_landmarks_on_image(self, image: np.ndarray, frame_result: FrameResult, 
                               show_crop_boxes: bool = True, show_merged_crop: bool = True,
                               square_crop: bool = True, padding: int = 50) -> np.ndarray:
        """
        在图像上绘制手部关键点和裁剪框
        
        Args:
            image: 输入图像
            frame_result: 检测结果
            show_crop_boxes: 是否显示单独的裁剪框
            show_merged_crop: 是否显示合并的裁剪框
            square_crop: 是否使用正方形裁剪
            padding: 边界填充
            
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
        
        # 绘制右手
        if frame_result.right_hand:
            self._draw_single_hand(
                annotated_image, 
                frame_result.right_hand, 
                color=(255, 0, 0),  # 蓝色
                label="Right Hand"
            )
        
        # 绘制裁剪框
        if show_crop_boxes or show_merged_crop:
            merged_crop_coords, crop_info = self.calculate_merged_crop_box(
                frame_result, padding=padding, square_crop=square_crop
            )
            
            if crop_info["crop_strategy"] != "none":
                # 绘制个别手的裁剪框
                if show_crop_boxes:
                    if "left_crop" in crop_info:
                        left_crop = np.array(crop_info["left_crop"])
                        self._draw_crop_box(annotated_image, left_crop, "Left Crop", (0, 255, 0))
                    
                    if "right_crop" in crop_info:
                        right_crop = np.array(crop_info["right_crop"])
                        self._draw_crop_box(annotated_image, right_crop, "Right Crop", (255, 0, 0))
                
                # 绘制合并的裁剪框
                if show_merged_crop:
                    strategy = crop_info["crop_strategy"]
                    if strategy == "merged_hands":
                        color = (0, 255, 255)  # 黄色
                        label = "Merged Crop"
                    else:
                        color = (255, 255, 0)  # 青色
                        label = "Single Crop"
                    
                    self._draw_crop_box(annotated_image, merged_crop_coords, label, color, thickness=3)
        
        # 添加帧信息
        info_text = f"Frame: {frame_result.frame_number}"
        if frame_result.left_hand or frame_result.right_hand:
            merged_crop_coords, crop_info = self.calculate_merged_crop_box(frame_result)
            info_text += f" | Strategy: {crop_info['crop_strategy']}"
        
        cv2.putText(
            annotated_image,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )
        
        return annotated_image
    
    def _draw_crop_box(self, image: np.ndarray, crop_coords: np.ndarray, 
                      label: str, color: Tuple[int, int, int], thickness: int = 2):
        """绘制裁剪框"""
        x1, y1, x2, y2 = crop_coords.astype(int)
        
        # 绘制裁剪框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # 添加标签
        cv2.putText(
            image,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            thickness
        )
    
    def _draw_single_hand(self, image: np.ndarray, hand_data: HandLandmarks, color: Tuple[int, int, int], label: str):
        """绘制单手关键点"""
        h, w = image.shape[:2]
        
        # 绘制关键点
        for i, (x, y, z) in enumerate(hand_data.landmarks):
            # 转换为像素坐标
            px, py = int(x * w), int(y * h)
            
            # 绘制关键点
            cv2.circle(image, (px, py), 3, color, -1)
            
            # 绘制关键点编号（较小）
            cv2.putText(
                image,
                str(i),
                (px + 3, py - 3),
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
                0.5,
                color,
                1
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
                cv2.line(image, start_point, end_point, color, 1)


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
        self.crops_info: List[Dict] = []  # 每帧的裁剪信息
    
    def process_video(self, video_path: str, output_video_path: Optional[str] = None, 
                     save_json: bool = True, save_hand_crops: bool = True, 
                     frame_skip: int = 1, show_merged_crop: bool = True,
                     square_crop: bool = True, padding: int = 50) -> bool:
        """
        处理视频文件
        
        Args:
            video_path: 输入视频路径
            output_video_path: 输出视频路径（可选）
            save_json: 是否保存JSON结果
            save_hand_crops: 是否保存手部裁剪图像
            frame_skip: 帧跳过间隔（1表示处理每一帧）
            show_merged_crop: 是否在输出视频中显示合并裁剪框
            square_crop: 是否生成正方形裁剪
            padding: 边界填充像素
            
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
                crop_info = {}
                if save_hand_crops and (frame_result.left_hand or frame_result.right_hand):
                    crop_info = self.detector.crop_hands_from_frame(
                        frame, frame_result, save_crops=True, output_dir=str(self.output_dir),
                        square_crop=square_crop, padding=padding
                    )
                self.crops_info.append(crop_info)
                
                # 绘制关键点和裁剪框
                annotated_frame = self.detector.draw_landmarks_on_image(
                    frame, frame_result, show_crop_boxes=False, show_merged_crop=show_merged_crop,
                    square_crop=square_crop, padding=padding
                )
                
                # 写入输出视频
                if video_writer:
                    video_writer.write(annotated_frame)
                
                processed_frames += 1
                
                # 显示进度
                if processed_frames % 30 == 0:
                    progress = (frame_number / total_frames) * 100
                    hands_count = 0
                    if frame_result.left_hand:
                        hands_count += 1
                    if frame_result.right_hand:
                        hands_count += 1
                    
                    strategy = crop_info.get('crop_strategy', 'none')
                    print(f"处理进度: {progress:.1f}% ({frame_number}/{total_frames}), "
                          f"当前帧检测到 {hands_count} 只手, 裁剪策略: {strategy}")
                
                frame_number += 1
            
            print(f"视频处理完成，共处理 {processed_frames} 帧")
            
            # 保存JSON结果
            if save_json:
                self._save_results_to_json(video_path, square_crop, padding)
            
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
        total_crops = len([c for c in self.crops_info if c.get('crop_strategy') != 'none'])
        single_hand_crops = len([c for c in self.crops_info if c.get('crop_strategy') == 'single_hand'])
        merged_crops = len([c for c in self.crops_info if c.get('crop_strategy') == 'merged_hands'])
        
        print(f"\n手部裁剪统计:")
        print(f"  总裁剪图像: {total_crops} 张")
        print(f"  单手裁剪: {single_hand_crops} 张")
        print(f"  双手合并裁剪: {merged_crops} 张")
        print(f"  保存路径: {self.output_dir}/hand_crops/")
    
    def _save_results_to_json(self, video_path: str, square_crop: bool, padding: int):
        """保存检测结果到JSON文件"""
        # 准备JSON数据
        json_data = {
            "video_info": {
                "source_path": video_path,
                "processed_time": datetime.now().isoformat(),
                "total_frames": len(self.frame_results),
                "processing_options": {
                    "square_crop": square_crop,
                    "padding": padding
                }
            },
            "detection_results": []
        }
        
        for i, frame_result in enumerate(self.frame_results):
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
            
            # 裁剪信息
            if i < len(self.crops_info):
                crop_info = self.crops_info[i].copy()
                # 移除不需要序列化的大对象
                if "cropped_image" in crop_info:
                    del crop_info["cropped_image"]
                frame_data["crop_info"] = crop_info
            
            json_data["detection_results"].append(frame_data)
        
        # 保存JSON文件
        video_name = Path(video_path).stem
        json_path = self.output_dir / f"{video_name}_hand_detection_merged_crops.json"
        
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
        
        # 裁剪策略统计
        single_hand_frames = len([c for c in self.crops_info if c.get('crop_strategy') == 'single_hand'])
        merged_hands_frames = len([c for c in self.crops_info if c.get('crop_strategy') == 'merged_hands'])
        
        return {
            "total_frames": total_frames,
            "frames_with_left_hand": frames_with_left_hand,
            "frames_with_right_hand": frames_with_right_hand,
            "frames_with_both_hands": frames_with_both_hands,
            "frames_with_any_hand": frames_with_any_hand,
            "single_hand_frames": single_hand_frames,
            "merged_hands_frames": merged_hands_frames,
            "left_hand_detection_rate": frames_with_left_hand / total_frames * 100,
            "right_hand_detection_rate": frames_with_right_hand / total_frames * 100,
            "both_hands_detection_rate": frames_with_both_hands / total_frames * 100,
            "any_hand_detection_rate": frames_with_any_hand / total_frames * 100,
            "merge_rate": merged_hands_frames / max(1, frames_with_any_hand) * 100
        }

    def create_cropped_video(self, video_path: str, output_cropped_video_path: str, 
                           square_crop: bool = True, padding: int = 50, 
                           default_size: Tuple[int, int] = (512, 512)) -> bool:
        """
        创建仅包含裁剪手部区域的视频
        
        Args:
            video_path: 输入视频路径
            output_cropped_video_path: 裁剪视频输出路径
            square_crop: 是否使用正方形裁剪
            padding: 边界填充
            default_size: 当没有检测到手时的默认帧大小
            
        Returns:
            是否成功创建裁剪视频
        """
        if not self.frame_results:
            print("错误：请先运行 process_video")
            return False
        
        # 打开原视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"错误：无法打开视频文件: {video_path}")
            return False
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # 确定输出视频尺寸（使用所有有效裁剪的最大尺寸）
        max_width = default_size[0]
        max_height = default_size[1]
        
        for crop_info in self.crops_info:
            if crop_info.get('crop_strategy') != 'none' and 'crop_size' in crop_info:
                w, h = crop_info['crop_size']
                max_width = max(max_width, w)
                max_height = max(max_height, h)
        
        # 如果使用正方形，取较大尺寸
        if square_crop:
            output_size = max(max_width, max_height)
            max_width = max_height = output_size
        
        print(f"裁剪视频尺寸: {max_width}x{max_height}")
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_cropped_video_path, fourcc, fps, (max_width, max_height))
        
        frame_idx = 0
        processed_crop_idx = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 查找对应的检测结果
                corresponding_result = None
                corresponding_crop = None
                
                for i, result in enumerate(self.frame_results):
                    if result.frame_number == frame_idx:
                        corresponding_result = result
                        if i < len(self.crops_info):
                            corresponding_crop = self.crops_info[i]
                        break
                
                # 创建输出帧
                if corresponding_crop and corresponding_crop.get('crop_strategy') != 'none':
                    # 重新计算裁剪区域
                    merged_crop_coords, _ = self.detector.calculate_merged_crop_box(
                        corresponding_result, padding=padding, square_crop=square_crop
                    )
                    
                    x1, y1, x2, y2 = merged_crop_coords.astype(int)
                    
                    if x2 > x1 and y2 > y1:
                        cropped_frame = frame[y1:y2, x1:x2]
                        
                        # 调整到目标尺寸
                        if cropped_frame.shape[1] != max_width or cropped_frame.shape[0] != max_height:
                            cropped_frame = cv2.resize(cropped_frame, (max_width, max_height))
                        
                        output_frame = cropped_frame
                    else:
                        # 创建黑色帧
                        output_frame = np.zeros((max_height, max_width, 3), dtype=np.uint8)
                else:
                    # 没有检测到手，创建黑色帧
                    output_frame = np.zeros((max_height, max_width, 3), dtype=np.uint8)
                    
                    # 添加"No hands detected"文本
                    cv2.putText(
                        output_frame,
                        "No hands detected",
                        (max_width//4, max_height//2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2
                    )
                
                video_writer.write(output_frame)
                frame_idx += 1
            
            print(f"裁剪视频创建完成: {output_cropped_video_path}")
            return True
            
        except Exception as e:
            print(f"创建裁剪视频时出错: {e}")
            return False
            
        finally:
            cap.release()
            video_writer.release()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='MediaPipe手部检测视频处理工具（支持双手合并裁剪）')
    parser.add_argument('video_path', help='输入视频文件路径（MP4格式）')
    parser.add_argument('--output-dir', default='output', help='输出目录')
    parser.add_argument('--output-video', help='输出视频文件路径（可选）')
    parser.add_argument('--output-cropped-video', help='裁剪视频输出路径（可选）')
    parser.add_argument('--no-json', action='store_true', help='不保存JSON结果')
    parser.add_argument('--no-crops', action='store_true', help='不保存手部裁剪图像')
    parser.add_argument('--no-square', action='store_true', help='不强制正方形裁剪')
    parser.add_argument('--frame-skip', type=int, default=1, help='帧跳过间隔（默认处理每一帧）')
    parser.add_argument('--padding', type=int, default=50, help='裁剪时的边界填充像素（默认50）')
    parser.add_argument('--default-crop-size', type=int, nargs=2, default=[512, 512], 
                       help='默认裁剪尺寸 (宽度 高度)，默认512x512')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.video_path):
        print(f"错误：输入视频文件不存在: {args.video_path}")
        return
    
    # 生成默认输出视频路径
    video_name = Path(args.video_path).stem
    if not args.output_video:
        args.output_video = os.path.join(args.output_dir, f"{video_name}_with_merged_crops.mp4")
    
    if not args.output_cropped_video:
        args.output_cropped_video = os.path.join(args.output_dir, f"{video_name}_cropped_hands.mp4")
    
    # 创建视频处理器
    processor = VideoProcessor(args.output_dir)
    
    # 处理视频
    success = processor.process_video(
        video_path=args.video_path,
        output_video_path=args.output_video,
        save_json=not args.no_json,
        save_hand_crops=not args.no_crops,
        frame_skip=args.frame_skip,
        show_merged_crop=True,
        square_crop=not args.no_square,
        padding=args.padding
    )
    
    if success:
        # 创建裁剪视频
        crop_success = processor.create_cropped_video(
            video_path=args.video_path,
            output_cropped_video_path=args.output_cropped_video,
            square_crop=not args.no_square,
            padding=args.padding,
            default_size=tuple(args.default_crop_size)
        )
        
        print(f"\n处理成功！输出文件:")
        print(f"  原视频（带标注）: {args.output_video}")
        if crop_success:
            print(f"  裁剪视频: {args.output_cropped_video}")
        if not args.no_json:
            json_path = os.path.join(args.output_dir, f"{video_name}_hand_detection_merged_crops.json")
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
        print(f"  单手裁剪帧数: {stats['single_hand_frames']}")
        print(f"  双手合并裁剪帧数: {stats['merged_hands_frames']}")
        print(f"  合并裁剪比例: {stats['merge_rate']:.1f}%")
    else:
        print("处理失败！")


if __name__ == "__main__":
    main()