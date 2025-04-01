"""
手影识别系统主程序
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
        self.consecutive_matches = 0
        self.last_match_id = None
        
        # 初始化组件
        self.init_components()
        
        # 处理线程
        self.processing_thread = None
        
        # 设置信号处理，优雅地处理退出
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        self.logger.info("System initialized")
    
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
            lambda cmd, addr: self.start_processing()
        )
        
        self.command_handler.register_handler(
            COMMAND_TYPES["STOP"], 
            lambda cmd, addr: self.stop_processing()
        )
        
        self.command_handler.register_handler(
            COMMAND_TYPES["SWITCH_SCENE"], 
            lambda cmd, addr: self.switch_group(cmd.get("scene_id"))
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
                
                # 重置匹配状态
                self.consecutive_matches = 0
                self.last_match_id = None
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
        
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
            self.processing_thread = None
        self.logger.info("Video processing stopped")
    
    @measure_fps
    def process_frame(self, frame):
        """
        处理单帧视频
        
        参数:
            frame (numpy.ndarray): 视频帧
            
        返回:
            numpy.ndarray: 处理后的帧
        """
        if frame is None:
            return None
        
        # 调整帧大小
        if frame.shape[1] != FRAME_WIDTH or frame.shape[0] != FRAME_HEIGHT:
            frame = resize_image(frame, FRAME_WIDTH, FRAME_HEIGHT)
        
        # 使用处理锁获取当前活动的模板组
        with self.processing_lock:
            current_templates = self.templates
            current_group = self.current_group
        
        try:
            # 检测手部
            mask, hand_contour = self.hand_detector.detect_hand(frame)
            
            # 创建信息显示区域
            info_panel = np.ones((frame.shape[0], 250, 3), dtype=np.uint8) * 240  # 浅灰色背景
            
            # 显示当前组信息
            cv.putText(
                info_panel,
                f"Group: {current_group}",
                (10, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),  # 黑色文本
                2
            )
            
            # 如果成功检测到手部轮廓
            all_match_results = []
            best_match = None
            
            if hand_contour is not None and current_templates:
                # 与模板比较
                match_result = self.contour_matcher.match_with_templates(hand_contour, current_templates)
                
                # 获取所有模板的匹配结果
                if len(current_templates) > 0:
                    # 为每个模板计算相似度
                    for template_id, template in current_templates.items():
                        template_contour = template.get("contour")
                        if template_contour is not None and hand_contour is not None:
                            similarity = self.contour_matcher._compare_contours(template_contour, hand_contour)
                            all_match_results.append({
                                "id": template_id,
                                "name": template.get("name", template_id),
                                "similarity": similarity,
                                "threshold": template.get("threshold", self.contour_matcher.default_threshold),
                                "matched": similarity > template.get("threshold", self.contour_matcher.default_threshold)
                            })
                    
                    # 按相似度排序
                    all_match_results.sort(key=lambda x: x["similarity"], reverse=True)
                
                # 处理匹配结果
                if match_result and match_result["matched"]:
                    best_match = match_result
                    # 检查是否与上一次匹配相同
                    if match_result["id"] == self.last_match_id:
                        self.consecutive_matches += 1
                    else:
                        self.consecutive_matches = 1
                        self.last_match_id = match_result["id"]
                    
                    # 在原始帧上显示最佳匹配结果
                    cv.putText(
                        frame,
                        f"Detected: {match_result['name']}",
                        (10, 30),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),  # 绿色
                        2
                    )
                    
                    cv.putText(
                        frame,
                        f"Similarity: {match_result['similarity']:.1f}%",
                        (10, 60),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),  # 绿色
                        2
                    )
                    
                    # 绘制轮廓
                    cv.drawContours(frame, [hand_contour], -1, (0, 255, 0), 2)
                    
                    # 连续匹配达到阈值，发送结果
                    if self.consecutive_matches >= CONSECUTIVE_FRAMES:
                        result_data = {
                            "id": match_result["id"],
                            "name": match_result["name"],
                            "similarity": match_result["similarity"],
                            "group": current_group
                        }
                        
                        # 发送识别结果
                        send_result(result_data, self.args.send_ip, self.args.send_port)
                        
                        # 日志记录
                        self.logger.info(f"Match found: {match_result['name']} "
                                        f"(similarity: {match_result['similarity']:.1f})")
                        
                        # 重置计数器，避免重复发送
                        self.consecutive_matches = 0
                else:
                    self.consecutive_matches = 0
                    self.last_match_id = None
                    
                    # 没有匹配，但仍然绘制轮廓
                    if hand_contour is not None:
                        cv.drawContours(frame, [hand_contour], -1, (0, 165, 255), 2)  # 橙色
                        
                        # 显示未匹配提示
                        cv.putText(
                            frame,
                            "No Match",
                            (10, 30),
                            cv.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 165, 255),  # 橙色
                            2
                        )
            
            # 在信息面板上显示所有模板的匹配情况
            cv.putText(
                info_panel,
                "Template Matching Results:",
                (10, 70),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),  # 黑色
                1
            )
            
            # 显示每个模板的匹配情况
            y_offset = 100
            for i, result in enumerate(all_match_results):
                # 确定文本颜色 - 最佳匹配为绿色，其他为黑色
                color = (0, 128, 0) if result == best_match else (0, 0, 0)  
                # 超过阈值的匹配显示绿色，否则显示红色
                match_color = (0, 128, 0) if result["matched"] else (0, 0, 128)
                
                cv.putText(
                    info_panel,
                    f"{i+1}. {result['name']}:",
                    (10, y_offset),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1
                )
                
                cv.putText(
                    info_panel,
                    f"{result['similarity']:.1f}% ({result['threshold']}%)",
                    (150, y_offset),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    match_color,
                    1
                )
                
                y_offset += 25
                
                # 防止显示超出边界
                if y_offset > frame.shape[0] - 30:
                    break
            
            # 显示连续匹配计数
            if self.consecutive_matches > 0:
                cv.putText(
                    info_panel,
                    f"Consecutive: {self.consecutive_matches}/{CONSECUTIVE_FRAMES}",
                    (10, frame.shape[0] - 30),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 128),  # 红色
                    1
                )
            
            # 显示当前组信息在原始帧上
            cv.putText(
                frame,
                f"Group: {current_group}",
                (10, frame.shape[0] - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            
            # 合并原始帧和信息面板
            combined_frame = np.hstack((frame, info_panel))
            
            return combined_frame
            
        except ZeroDivisionError as e:
            # 专门处理除零错误，提供更具体的信息
            self.logger.error(f"Division by zero error in processing frame: {e}")
            # 返回带有错误信息的原始帧
            cv.putText(
                frame,
                "Error: Division by zero detected",
                (50, 50),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
            return frame
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")
            return frame
    
    def process_video_loop(self):
        """视频处理主循环"""
        # 确定视频源
        video_source = self.args.video if self.args.video else self.args.camera
        self.logger.info(f"Opening video source: {video_source}")
        
        # 打开视频源
        cap = cv.VideoCapture(video_source)
        if not cap.isOpened():
            self.logger.error(f"Failed to open video source: {video_source}")
            return
        
        # 获取视频信息
        frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv.CAP_PROP_FPS))
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT)) if self.args.video else 0
        
        self.logger.info(f"Video info: {frame_width}x{frame_height} @ {fps}fps")
        if total_frames > 0:
            self.logger.info(f"Total frames: {total_frames}")
        
        # 帧计数
        frame_count = 0
        
        try:
            # 主循环
            while self.running and cap.isOpened():
                ret, frame = cap.read()
                frame_count += 1
                
                if not ret:
                    self.logger.info("End of video reached")
                    break
                
                # 帧率控制
                if frame_count % self.args.skip != 0:
                    continue
                
                # 处理帧
                processed_frame = self.process_frame(frame)
                
                # 显示结果
                if self.args.show and processed_frame is not None:
                    # 调整窗口大小以适应更宽的帧
                    cv.namedWindow('Hand Shadow Recognition', cv.WINDOW_NORMAL)
                    # 计算适当的窗口大小
                    if processed_frame.shape[1] > FRAME_WIDTH:
                        display_width = min(1280, processed_frame.shape[1])  # 限制最大宽度
                        display_height = int(display_width * processed_frame.shape[0] / processed_frame.shape[1])
                        cv.resizeWindow('Hand Shadow Recognition', display_width, display_height)
                    
                    cv.imshow('Hand Shadow Recognition', processed_frame)
                    if cv.waitKey(1) & 0xFF == ord('q'):
                        break
            
        except Exception as e:
            self.logger.error(f"Error in video processing loop: {e}")
            
        finally:
            # 释放资源
            cap.release()
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