"""
手影识别系统主程序 - 支持双视频流
整合各个模块，实现完整功能
"""

import os
import sys
import time
import argparse
import threading
import cv2 as cv
import signal
import numpy as np

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 

from handpose_shadow.config import (
    VIDEO_SOURCE, FRAME_SKIP, CONSECUTIVE_FRAMES, FRAME_WIDTH, FRAME_HEIGHT,
    UDP_SEND_IP, UDP_SEND_PORT, UDP_LISTEN_IP, UDP_LISTEN_PORT,
    DEFAULT_GROUP, COMMAND_TYPES, SHOW_PREVIEW
)
from handpose_shadow.template_manager import TemplateManager
from handpose_shadow.hand_detector import HandDetector
from handpose_shadow.contour_matcher import ContourMatcher
from handpose_shadow.network_utils import send_result, send_ok, send_error
from handpose_shadow.command_server import CommandServer, CommandHandler
from handpose_shadow.utils.logging_utils import get_logger
from handpose_shadow.utils import resize_image, measure_fps

class HandShadowSystem:
    """手影识别系统主类，协调各个模块工作"""
    
    def __init__(self):
        """初始化系统"""
        self.logger = get_logger("main_system")
        self.logger.info("Initializing hand shadow recognition system")
        
        # 解析命令行参数
        self.args = self.parse_args()
        
        # 系统状态
        self.running = False
        self.current_group = self.args.group
        self.processing_lock = threading.Lock()
        
        # 初始化组件
        self.init_components()
        
        # 处理线程
        self.processing_thread = None
        
        # 设置信号处理，优雅地处理退出
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        # 双流控制状态
        self.left_stream_active = False    
        self.right_stream_active = False   
        self.left_target_id = "1001" # 设置默认为 1001         
        self.right_target_id = "1002"   # 设置默认为 1001     
        
        # 双流状态锁
        self.stream_control_lock = threading.Lock()
        
        self.logger.info("System initialized")
    
    def activate_left_stream(self, target_id):
        """激活左流识别并发送特定目标的结果"""
        with self.stream_control_lock:
            self.left_stream_active = True
            self.left_target_id = target_id
            self.logger.info(f"Left stream result sending activated for target: {target_id}")

    def activate_right_stream(self, target_id):
        """激活右流识别并发送特定目标的结果"""
        with self.stream_control_lock:
            self.right_stream_active = True
            self.right_target_id = target_id
            self.logger.info(f"Right stream result sending activated for target: {target_id}")

    def parse_args(self):
        """解析命令行参数"""
        parser = argparse.ArgumentParser(description='手影识别系统')
        parser.add_argument('--video', type=str, help='视频文件路径')
        parser.add_argument('--camera', type=int, default=VIDEO_SOURCE, help='摄像头ID')
        parser.add_argument('--group', type=str, default=DEFAULT_GROUP, help='初始模板组')
        parser.add_argument('--send-ip', type=str, default=UDP_SEND_IP, help='结果发送目标IP')
        parser.add_argument('--send-port', type=int, default=UDP_SEND_PORT, help='结果发送目标端口')
        parser.add_argument('--listen-ip', type=str, default=UDP_LISTEN_IP, help='命令监听IP')
        parser.add_argument('--listen-port', type=int, default=UDP_LISTEN_PORT, help='命令监听端口')
        parser.add_argument('--skip', type=int, default=FRAME_SKIP, help='帧跳过数')
        parser.add_argument('--show', action='store_true', default=SHOW_PREVIEW, help='显示预览窗口')
        parser.add_argument('--debug', action='store_true', help='启用调试模式')
        return parser.parse_args()
    
    def init_components(self):
        """初始化系统组件"""
        # 创建模板管理器
        self.template_manager = TemplateManager()
        
        # 创建手部检测器
        self.hand_detector = HandDetector()
        
        # 创建轮廓匹配器
        self.contour_matcher = ContourMatcher()
        
        # 创建命令处理器
        self.command_handler = CommandHandler()
        self.register_command_handlers()
        
        # 创建命令服务器
        self.command_server = CommandServer(
            listen_ip=self.args.listen_ip,
            listen_port=self.args.listen_port,
            callback=self.handle_command
        )
        
        # 加载初始模板组
        self.templates = self.template_manager.get_group(self.current_group)
        self.logger.info(f"Loaded {len(self.templates)} templates from group '{self.current_group}'")
    
    def register_command_handlers(self):
        """注册命令处理函数"""
        self.command_handler.register_handler(
            COMMAND_TYPES["START"], 
            lambda cmd, addr: self.start_processing()  # 只启动处理，不激活结果发送
        )
        
        self.command_handler.register_handler(
            COMMAND_TYPES["STOP"], 
            lambda cmd, addr: self.stop_processing()   # 停止处理和所有结果发送
        )
        
        self.command_handler.register_handler(
            COMMAND_TYPES["SWITCH_SCENE"], 
            lambda cmd, addr: self.switch_group(cmd.get("scene_id"))
        )
        
        self.command_handler.register_handler(
            COMMAND_TYPES["LEFT"], 
            lambda cmd, addr: self.activate_left_stream(cmd.get("scene_id"))
        )
        
        self.command_handler.register_handler(
            COMMAND_TYPES["RIGHT"], 
            lambda cmd, addr: self.activate_right_stream(cmd.get("scene_id"))
        )
        
        self.command_handler.register_handler(
            COMMAND_TYPES["PING"], 
            lambda cmd, addr: self.logger.debug(f"Received ping from {addr}")
        )

    def handle_command(self, command, addr):
        """
        处理接收到的命令
        
        参数:
            command (dict): 命令数据
            addr (tuple): 发送方地址
        """
        self.logger.info(f"Received command: {command['type']} from {addr[0]}:{addr[1]}")
        self.command_handler.handle_command(command, addr)
    
    def switch_group(self, group_id):
        """
        切换当前活动的模板组
        
        参数:
            group_id (str): 模板组ID
        """
        if not group_id:
            self.logger.warning("Cannot switch group: no group_id provided")
            return
        
        with self.processing_lock:
            try:
                self.templates = self.template_manager.get_group(group_id)
                self.current_group = group_id
                self.logger.info(f"Switched to template group: {group_id}, "
                                f"loaded {len(self.templates)} templates")
            except Exception as e:
                self.logger.error(f"Error switching template group: {e}")
    
    def start_processing(self):
        """启动视频处理"""
        if self.running:
            self.logger.warning("System is already running")
            return
        
        self.running = True
        self.processing_thread = threading.Thread(target=self.process_video_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        self.logger.info("Video processing started")
    
    def stop_processing(self):
        """停止视频处理"""
        if not self.running:
            self.logger.warning("System is not running")
            return
        
        # 停止处理线程
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
            self.processing_thread = None
        
        # 停止所有流的结果发送
        with self.stream_control_lock:
            self.left_stream_active = False
            self.right_stream_active = False
            self.left_target_id = None
            self.right_target_id = None
        
        self.logger.info("Video processing stopped, all stream result sending deactivated")

    
    def detect_and_init_video_sources(self):
        """检测并初始化视频源"""
        video_sources = []
        
        # 方案1: 如果指定了视频文件，只使用视频文件
        if self.args.video:
            cap = cv.VideoCapture(self.args.video)
            if cap.isOpened():
                video_sources.append(cap)
                self.logger.info(f"Using video file: {self.args.video}")
            else:
                self.logger.error(f"Failed to open video file: {self.args.video}")
            return video_sources
        
        # 方案2: 检测可用的摄像头
        self.logger.info("Detecting available cameras...")
        
        # 检测摄像头0
        cap0 = cv.VideoCapture(0)
        if cap0.isOpened():
            # 简单测试是否真的能读取帧
            ret, _ = cap0.read()
            if ret:
                video_sources.append(cap0)
                self.logger.info("Camera 0 detected and added")
            else:
                self.logger.warning("Camera 0 detected but cannot read frames")
                cap0.release()
        else:
            self.logger.warning("Camera 0 not available")
            cap0.release()
        
        # 检测摄像头1
        cap1 = cv.VideoCapture(1)
        if cap1.isOpened():
            # 简单测试是否真的能读取帧
            ret, _ = cap1.read()
            if ret:
                video_sources.append(cap1)
                self.logger.info("Camera 1 detected and added")
            else:
                self.logger.warning("Camera 1 detected but cannot read frames")
                cap1.release()
        else:
            self.logger.info("Camera 1 not available")
            cap1.release()
        
        # 如果没有找到任何摄像头，尝试使用默认的camera参数
        if not video_sources:
            self.logger.warning("No cameras auto-detected, trying default camera...")
            cap_default = cv.VideoCapture(self.args.camera)
            if cap_default.isOpened():
                ret, _ = cap_default.read()
                if ret:
                    video_sources.append(cap_default)
                    self.logger.info(f"Using default camera: {self.args.camera}")
                else:
                    cap_default.release()
            else:
                cap_default.release()
        
        return video_sources
    
    def process_single_stream_frame(self, frame, stream_id, state):
        """
        处理单个视频流的帧 - 简化版本，只匹配target_id模板
        
        参数:
            frame: 输入帧
            stream_id: 流标识 ("stream_0" 或 "stream_1")
            state: 流状态字典
            
        返回:
            处理后的帧
        """
        if frame is None:
            return None

        # 调整帧大小
        if frame.shape[1] != FRAME_WIDTH or frame.shape[0] != FRAME_HEIGHT:
            frame = resize_image(frame, FRAME_WIDTH, FRAME_HEIGHT)

        # 获取当前模板组
        with self.processing_lock:
            current_templates = self.templates
            current_group = self.current_group

        # 获取当前流的控制状态
        with self.stream_control_lock:
            if stream_id == "stream_0":  # 左流
                should_send_result = self.left_stream_active
                target_id = self.left_target_id
                stream_type = "left"
            elif stream_id == "stream_1":  # 右流
                should_send_result = self.right_stream_active
                target_id = self.right_target_id
                stream_type = "right"
            else:
                return frame  # 无效流ID直接返回
        
        # 如果没有target_id或target_id不在模板中，直接返回
        if not target_id or target_id not in current_templates:
            cv.putText(frame, f"No target set for {stream_id}", (10, 30), 
                    cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return frame

        try:
            # 检测手部
            mask, hand_contour = self.hand_detector.detect_hand(frame)
            
            # 创建信息显示区域
            info_panel = np.ones((frame.shape[0], 300, 3), dtype=np.uint8) * 240
            
            # 显示流信息
            cv.putText(info_panel, f"Stream: {stream_id}", (10, 30), 
                    cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 128), 2)
            cv.putText(info_panel, f"Group: {current_group}", (10, 55), 
                    cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # 显示激活状态
            status_text = f"Active: {should_send_result}"
            status_color = (0, 128, 0) if should_send_result else (0, 0, 128)
            cv.putText(info_panel, status_text, (10, 80), 
                    cv.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            
            # 显示目标ID
            cv.putText(info_panel, f"Target: {target_id}", (10, 105), 
                    cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            match_result = None

            if hand_contour is not None:
                # 只与target_id模板匹配
                target_template = current_templates[target_id]
                similarity = self.contour_matcher._compare_contours(
                    target_template["contour"], hand_contour)
                threshold = target_template.get("threshold", self.contour_matcher.default_threshold)
                
                match_result = {
                    "id": target_id,
                    "name": target_template.get("name", target_id),
                    "similarity": similarity,
                    "threshold": threshold,
                    "matched": similarity > threshold
                }

                # 在信息面板显示目标模板匹配情况
                cv.putText(info_panel, "Target Match:", (10, 140), 
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                similarity_color = (0, 128, 0) if match_result["matched"] else (0, 0, 128)
                cv.putText(info_panel, f"Similarity: {match_result['similarity']:.1f}%", (10, 165), 
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, similarity_color, 1)
                cv.putText(info_panel, f"Threshold: {match_result['threshold']:.1f}", (10, 185), 
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                cv.putText(info_panel, f"Matched: {match_result['matched']}", (10, 205), 
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, similarity_color, 1)

                # 处理匹配结果
                if match_result["matched"]:
                    # 连续匹配计数
                    if match_result["id"] == state['last_match_id']:
                        state['consecutive_matches'] += 1
                    else:
                        state['consecutive_matches'] = 1
                        state['last_match_id'] = match_result["id"]
                    
                    # 在原始帧上显示匹配结果
                    result_color = (0, 255, 0)
                    cv.putText(frame, f"Detected: {match_result['name']}", (10, 30), 
                            cv.FONT_HERSHEY_SIMPLEX, 0.8, result_color, 2)
                    cv.putText(frame, f"Similarity: {match_result['similarity']:.1f}%", (10, 60), 
                            cv.FONT_HERSHEY_SIMPLEX, 0.6, result_color, 2)
                    
                    # 绘制轮廓
                    cv.drawContours(frame, [hand_contour], -1, result_color, 2)
                    
                    # 关键逻辑：只有在流激活且连续匹配达到阈值时才发送结果
                    if (should_send_result and 
                        state['consecutive_matches'] >= CONSECUTIVE_FRAMES):
                        
                        result_data = {
                            "id": match_result["id"],
                            "name": match_result["name"],
                            "similarity": match_result["similarity"],
                            "group": current_group,
                            "stream_id": stream_id,
                            "stream_type": stream_type
                        }
                        
                        # 发送识别结果
                        send_result(result_data, self.args.send_ip, self.args.send_port)
                        
                        self.logger.info(f"[{stream_type}] Match sent: {match_result['name']} "
                                        f"(similarity: {match_result['similarity']:.1f})")
                        
                        # 重置计数器，避免重复发送
                        state['consecutive_matches'] = 0
                else:
                    # 匹配失败，重置状态
                    state['consecutive_matches'] = 0
                    state['last_match_id'] = None
                    
                    # 绘制检测到的轮廓但显示未匹配
                    cv.drawContours(frame, [hand_contour], -1, (0, 165, 255), 2)
                    cv.putText(frame, f"No Match (target: {target_id})", (10, 30), 
                            cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            else:
                # 没有检测到手部
                state['consecutive_matches'] = 0
                state['last_match_id'] = None
                cv.putText(frame, "No Hand Detected", (10, 30), 
                        cv.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)

            # 显示连续匹配计数
            if state['consecutive_matches'] > 0:
                cv.putText(info_panel, f"Consecutive: {state['consecutive_matches']}/{CONSECUTIVE_FRAMES}", 
                        (10, frame.shape[0] - 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 128), 1)

            # 合并原始帧和信息面板
            combined_frame = np.hstack((frame, info_panel))
            return combined_frame

        except Exception as e:
            self.logger.error(f"Error processing frame in {stream_id}: {e}")
            cv.putText(frame, f"Error in {stream_id}", (50, 50), 
                    cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame
        
    def display_frames(self, processed_frames):
        """显示处理后的帧"""
        if len(processed_frames) == 1:
            # 单视频流 - 直接显示
            stream_id, frame = processed_frames[0]
            cv.imshow('Hand Shadow Recognition', frame)
            
        elif len(processed_frames) == 2:
            # 双视频流 - 垂直拼接显示
            stream1_id, frame1 = processed_frames[0]
            stream2_id, frame2 = processed_frames[1]
            
            # 确保两个帧宽度相同
            h1, w1 = frame1.shape[:2]
            h2, w2 = frame2.shape[:2]
            
            if w1 != w2:
                # 调整到相同宽度
                target_width = min(w1, w2)
                if w1 != target_width:
                    frame1 = cv.resize(frame1, (target_width, int(h1 * target_width / w1)))
                if w2 != target_width:
                    frame2 = cv.resize(frame2, (target_width, int(h2 * target_width / w2)))
            
            # 垂直拼接
            combined_frame = np.vstack((frame1, frame2))
            
            # 在拼接线处添加分隔线
            h, w = combined_frame.shape[:2]
            cv.line(combined_frame, (0, h//2), (w, h//2), (255, 255, 255), 3)
            
            # 显示
            cv.namedWindow('Hand Shadow Recognition - Dual Stream', cv.WINDOW_NORMAL)
            # 设置窗口大小
            display_width = min(1200, w)
            display_height = int(display_width * h / w)
            cv.resizeWindow('Hand Shadow Recognition - Dual Stream', display_width, display_height)
            cv.imshow('Hand Shadow Recognition - Dual Stream', combined_frame)
        
        else:
            # 多于2个流 - 分别显示
            for stream_id, frame in processed_frames:
                window_name = f'Hand Shadow - {stream_id}'
                cv.imshow(window_name, frame)
    
    def process_video_loop(self):
        """视频处理主循环 - 支持单/双视频流"""
        
        # 检测并初始化视频源
        video_sources = self.detect_and_init_video_sources()
        if not video_sources:
            self.logger.error("No valid video sources found")
            return
            
        self.logger.info(f"Processing {len(video_sources)} video stream(s)")
        
        # 为每个视频流初始化状态
        stream_states = {}
        for i, cap in enumerate(video_sources):
            stream_id = f"stream_{i}"
            stream_states[stream_id] = {
                'cap': cap,
                'frame_count': 0,
                'consecutive_matches': 0,
                'last_match_id': None
            }
            
            # 获取视频信息
            width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv.CAP_PROP_FPS))
            total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT)) if hasattr(cap, 'get') else 0
            self.logger.info(f"{stream_id}: {width}x{height} @ {fps}fps")
            if total_frames > 0:
                self.logger.info(f"{stream_id}: Total frames: {total_frames}")
        
        try:
            # 主循环 - 单线程轮询处理
            while self.running:
                all_streams_ended = True
                processed_frames = []
                
                # 依次处理每个视频流
                for stream_id, state in stream_states.items():
                    cap = state['cap']
                    
                    if not cap.isOpened():
                        continue
                        
                    # 读取一帧
                    ret, frame = cap.read()
                    if not ret:
                        self.logger.info(f"{stream_id}: End of video reached")
                        continue
                        
                    all_streams_ended = False
                    state['frame_count'] += 1
                    
                    # 帧率控制
                    if state['frame_count'] % self.args.skip != 0:
                        continue
                    
                    # 处理帧
                    processed_frame = self.process_single_stream_frame(
                        frame, stream_id, state
                    )
                    
                    if processed_frame is not None:
                        processed_frames.append((stream_id, processed_frame))
                
                # 如果所有流都结束了，退出循环
                if all_streams_ended:
                    self.logger.info("All video streams ended")
                    break
                
                # 显示处理结果
                if self.args.show and processed_frames:
                    self.display_frames(processed_frames)
                    if cv.waitKey(1) & 0xFF == ord('q'):
                        break
                        
        except Exception as e:
            self.logger.error(f"Error in video processing loop: {e}")
            
        finally:
            # 释放资源
            for state in stream_states.values():
                state['cap'].release()
            if self.args.show:
                cv.destroyAllWindows()
            
            self.logger.info("Video processing loop ended")
            self.running = False
    
    def signal_handler(self, sig, frame):
        """处理SIGINT和SIGTERM信号"""
        self.logger.info(f"Received signal {sig}, shutting down...")
        self.stop()
        sys.exit(0)
    
    def start_command_server(self):
        """启动命令服务器"""
        result = self.command_server.start()
        if result:
            self.logger.info(f"Command server started on {self.args.listen_ip}:{self.args.listen_port}")
        else:
            self.logger.error("Failed to start command server")
    
    def stop_command_server(self):
        """停止命令服务器"""
        result = self.command_server.stop()
        if result:
            self.logger.info("Command server stopped")
        else:
            self.logger.error("Failed to stop command server")
    
    def start(self):
        """启动系统"""
        # 启动命令服务器
        self.start_command_server()
        
        # 如果设置了自动启动，则开始处理
        if self.args.video or self.args.show:
            self.start_processing()
        
        self.logger.info("System started")
    
    def stop(self):
        """停止系统"""
        # 停止视频处理
        self.stop_processing()
        
        # 停止命令服务器
        self.stop_command_server()
        
        self.logger.info("System stopped")
    
    def run(self):
        """运行系统，直到退出"""
        try:
            # 启动系统
            self.start()
            
            # 主线程等待
            self.logger.info("System running, press Ctrl+C to exit...")
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received")
        finally:
            # 清理资源
            self.stop()
            self.logger.info("System shutdown complete")

def main():
    """主函数"""
    system = HandShadowSystem()
    system.run()

if __name__ == "__main__":
    main()