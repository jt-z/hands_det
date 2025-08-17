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
        self.finger_mcps = [2, 5, 9, 13, 17]
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
    
    def draw_landmarks_on_image(self, image: np.ndarray, frame_result: FrameResult) -> np.ndarray:
        """
        在图像上绘制手部关键点
        
        Args:
            image: 输入图像
            frame_result: 检测结果
            
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
    
    def process_video(self, video_path: str, output_video_path: Optional[str] = None, 
                     save_json: bool = True, frame_skip: int = 1) -> bool:
        """
        处理视频文件
        
        Args:
            video_path: 输入视频路径
            output_video_path: 输出视频路径（可选）
            save_json: 是否保存JSON结果
            frame_skip: 帧跳过间隔（1表示处理每一帧）
            
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
                
                # 绘制关键点
                annotated_frame = self.detector.draw_landmarks_on_image(frame, frame_result)
                
                # 写入输出视频
                if video_writer:
                    video_writer.write(annotated_frame)
                
                processed_frames += 1
                
                # 显示进度
                if processed_frames % 30 == 0:
                    progress = (frame_number / total_frames) * 100
                    print(f"处理进度: {progress:.1f}% ({frame_number}/{total_frames})")
                
                frame_number += 1
            
            print(f"视频处理完成，共处理 {processed_frames} 帧")
            
            # 保存JSON结果
            if save_json:
                self._save_results_to_json(video_path)
            
            return True
            
        except Exception as e:
            print(f"处理视频时出错: {e}")
            return False
            
        finally:
            # 释放资源
            cap.release()
            if video_writer:
                video_writer.release()
    
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
    parser.add_argument('--frame-skip', type=int, default=1, help='帧跳过间隔（默认处理每一帧）')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.video_path):
        print(f"错误：输入视频文件不存在: {args.video_path}")
        return
    
    # 生成默认输出视频路径
    if not args.output_video:
        video_name = Path(args.video_path).stem
        args.output_video = os.path.join(args.output_dir, f"{video_name}_with_landmarks.mp4")
    
    # 创建视频处理器
    processor = VideoProcessor(args.output_dir)
    
    # 处理视频
    success = processor.process_video(
        video_path=args.video_path,
        output_video_path=args.output_video,
        save_json=not args.no_json,
        frame_skip=args.frame_skip
    )
    
    if success:
        print(f"处理成功！输出文件:")
        print(f"  视频: {args.output_video}")
        if not args.no_json:
            video_name = Path(args.video_path).stem
            json_path = os.path.join(args.output_dir, f"{video_name}_hand_detection.json")
            print(f"  JSON: {json_path}")
        
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