"""
基于MediaPipe的手势匹配测试脚本
从视频中抽取帧并与模板图片进行手势相似度比较
"""

import os
import sys
import cv2 as cv
import numpy as np
import argparse
from pathlib import Path

# 添加项目路径到系统路径
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 

try:
    from handpose_shadow.image_matcher import ImageMatcher
    from handpose_shadow.config import TEMPLATES_DIR
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保在正确的项目目录下运行此脚本")
    print("需要安装 mediapipe: pip install mediapipe")
    sys.exit(1)

class HandGestureTester:
    """基于MediaPipe的手势匹配测试器"""
    
    def __init__(self, template_path, similarity_threshold=55):
        """
        初始化测试器
        
        参数:
            template_path (str): 模板图片路径
            similarity_threshold (float): 相似度阈值
        """
        self.template_path = template_path
        self.similarity_threshold = similarity_threshold
        
        # 初始化ImageMatcher
        self.image_matcher = ImageMatcher(default_threshold=similarity_threshold)
        
        # 加载模板图像
        self.template_image = self._load_template_image()
        if self.template_image is None:
            raise ValueError(f"无法加载模板图像: {template_path}")
        
        print(f"模板图像加载成功: {template_path}")
        print(f"模板图像尺寸: {self.template_image.shape}")
        print(f"相似度阈值: {similarity_threshold}")
        
        # 生成模板预览
        self._save_template_preview()
    
    def _load_template_image(self):
        """加载模板图像"""
        if not os.path.exists(self.template_path):
            print(f"模板文件不存在: {self.template_path}")
            return None
        
        try:
            template_img = cv.imread(self.template_path)
            if template_img is None:
                print(f"无法读取图像: {self.template_path}")
                return None
            
            return template_img
            
        except Exception as e:
            print(f"加载模板图像时出错: {e}")
            return None
    
    def _save_template_preview(self):
        """保存模板预览图，显示检测到的关键点"""
        try:
            # 提取模板关键点
            landmarks = self.image_matcher.extract_landmarks(self.template_image)
            
            if landmarks is not None:
                # 可视化关键点
                preview_img = self.image_matcher.visualize_landmarks(
                    self.template_image, landmarks, "Template Hand Landmarks"
                )
                
                # 添加信息文本
                h, w = preview_img.shape[:2]
                cv.rectangle(preview_img, (10, 10), (400, 100), (0, 0, 0), -1)
                cv.rectangle(preview_img, (10, 10), (400, 100), (255, 255, 255), 2)
                
                cv.putText(preview_img, "Template Preview", (20, 35), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv.putText(preview_img, f"Landmarks: {len(landmarks)} points", (20, 60), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv.putText(preview_img, f"Threshold: {self.similarity_threshold}", (20, 85), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # 保存预览图
                preview_path = "template_landmarks_preview.jpg"
                cv.imwrite(preview_path, preview_img)
                print(f"模板关键点预览已保存: {preview_path}")
            else:
                print("警告: 模板图像中未检测到手部关键点")
                # 保存原始模板图像作为预览
                cv.imwrite("template_original_preview.jpg", self.template_image)
                print("已保存原始模板图像预览: template_original_preview.jpg")
            
        except Exception as e:
            print(f"保存模板预览失败: {e}")
    
    def test_video(self, video_path, frame_skip=3, max_frames=None, save_results=False, output_video=True):
        """
        测试视频中的帧
        
        参数:
            video_path (str): 视频文件路径
            frame_skip (int): 每隔多少帧处理一次
            max_frames (int): 最大处理帧数
            save_results (bool): 是否保存结果图像
            output_video (bool): 是否输出结果视频
        """
        if not os.path.exists(video_path):
            print(f"视频文件不存在: {video_path}")
            return
        
        # 打开视频
        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频: {video_path}")
            return
        
        # 获取视频信息
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv.CAP_PROP_FPS)
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        
        print(f"\n视频信息:")
        print(f"  路径: {video_path}")
        print(f"  总帧数: {total_frames}")
        print(f"  帧率: {fps:.2f}")
        print(f"  分辨率: {width}x{height}")
        print(f"  处理策略: 每{frame_skip}帧处理1帧")
        
        # 创建结果保存目录
        if save_results:
            results_dir = "test_results"
            os.makedirs(results_dir, exist_ok=True)
        
        # 设置输出视频
        video_writer = None
        output_video_path = None
        if output_video:
            # 生成输出视频文件名
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_video_path = f"output_{video_name}_gesture_test.mp4"
            
            # 尝试不同的编码器
            fourccs = [
                ('mp4v', 'MP4V'),
                ('XVID', 'XVID'), 
                ('MJPG', 'MJPG'),
                ('X264', 'X264')
            ]
            
            for fourcc_str, fourcc_name in fourccs:
                try:
                    fourcc = cv.VideoWriter_fourcc(*fourcc_str)
                    video_writer = cv.VideoWriter(output_video_path, fourcc, max(1, fps/frame_skip), (width, height))
                    
                    if video_writer.isOpened():
                        print(f"  输出视频: {output_video_path} (编码器: {fourcc_name})")
                        break
                    else:
                        video_writer.release()
                        video_writer = None
                except:
                    continue
            
            if video_writer is None:
                print("警告: 尝试所有编码器都失败，将不生成输出视频")
                output_video = False
        
        # 统计变量
        frame_count = 0
        processed_count = 0
        match_count = 0
        similarities = []
        landmarks_detected_count = 0
        
        print(f"\n开始处理视频...")
        print("-" * 80)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 跳帧处理
                if frame_count % frame_skip != 0:
                    continue
                
                processed_count += 1
                
                # 检查最大帧数限制
                if max_frames and processed_count > max_frames:
                    break
                
                # 处理当前帧
                result = self._process_frame(frame, frame_count)
                
                # 创建可视化结果帧
                if result:
                    similarity = result['similarity']
                    matched = result['matched']
                    has_landmarks = result['has_landmarks']
                    
                    if has_landmarks:
                        landmarks_detected_count += 1
                        similarities.append(similarity)
                        
                        if matched:
                            match_count += 1
                    
                    # 打印结果
                    if has_landmarks:
                        status = "✓ 匹配" if matched else "✗ 不匹配"
                        print(f"帧 {frame_count:6d}: 相似度 {similarity:6.2f} - {status}")
                    else:
                        print(f"帧 {frame_count:6d}: 未检测到手部关键点")
                    
                    # 创建可视化帧
                    result_frame = self._visualize_result(frame, result)
                    
                    # 保存结果图像
                    if save_results and has_landmarks:
                        output_path = os.path.join(results_dir, f"frame_{frame_count:06d}_sim_{similarity:.1f}.jpg")
                        cv.imwrite(output_path, result_frame)
                
                else:
                    print(f"帧 {frame_count:6d}: 处理失败")
                    result_frame = self._visualize_no_detection(frame, frame_count)
                
                # 写入视频帧
                if video_writer is not None:
                    # 确保帧的尺寸正确
                    if result_frame.shape[:2] != (height, width):
                        result_frame = cv.resize(result_frame, (width, height))
                    video_writer.write(result_frame)
        
        except KeyboardInterrupt:
            print("\n用户中断处理")
        
        finally:
            cap.release()
            # 释放视频写入器
            if video_writer is not None:
                video_writer.release()
                if output_video_path and os.path.exists(output_video_path):
                    file_size = os.path.getsize(output_video_path) / (1024 * 1024)  # MB
                    print(f"\n✓ 结果视频已保存: {output_video_path} ({file_size:.1f} MB)")
                else:
                    print(f"\n✗ 视频保存失败: {output_video_path}")
        
        # 输出统计结果
        self._print_statistics(processed_count, landmarks_detected_count, match_count, similarities)
    
    def _process_frame(self, frame, frame_number):
        """处理单帧"""
        try:
            # 使用ImageMatcher进行手势匹配
            template_name = os.path.splitext(os.path.basename(self.template_path))[0]
            match_result = self.image_matcher.match_with_template(
                self.template_image, 
                frame, 
                template_name, 
                self.similarity_threshold
            )
            
            # 检查是否检测到手部关键点
            frame_landmarks = self.image_matcher.extract_landmarks(frame)
            has_landmarks = frame_landmarks is not None
            
            # 调试信息
            if frame_number % 30 == 0 and has_landmarks:
                print(f"  调试 - 帧 {frame_number}: 检测到手部关键点")
            
            return {
                'frame_number': frame_number,
                'similarity': match_result.get('similarity', 0.0),
                'matched': match_result.get('matched', False),
                'has_landmarks': has_landmarks,
                'frame_landmarks': frame_landmarks,
                'match_result': match_result
            }
            
        except Exception as e:
            print(f"处理帧 {frame_number} 时出错: {e}")
            if frame_number % 100 == 0:  # 每100帧打印一次详细错误
                import traceback
                traceback.print_exc()
            return None
    
    def _visualize_result(self, frame, result):
        """可视化检测结果"""
        result_frame = frame.copy()
        
        similarity = result['similarity']
        matched = result['matched']
        frame_number = result['frame_number']
        has_landmarks = result['has_landmarks']
        frame_landmarks = result['frame_landmarks']
        
        # 如果检测到关键点，绘制关键点
        if has_landmarks and frame_landmarks is not None:
            result_frame = self.image_matcher.visualize_landmarks(
                result_frame, frame_landmarks, "Hand Detection"
            )
        
        # 绘制模板缩略图（右上角）
        h_frame, w_frame = frame.shape[:2]
        template_size = min(w_frame, h_frame) // 4
        
        # 调整模板图像大小
        template_resized = cv.resize(self.template_image, (template_size, template_size))
        
        # 放置在右上角
        result_frame[10:10+template_size, w_frame-template_size-10:w_frame-10] = template_resized
        
        # 绘制模板边框
        cv.rectangle(result_frame, 
                    (w_frame-template_size-10, 10), 
                    (w_frame-10, 10+template_size), 
                    (255, 255, 255), 2)
        
        # 绘制主要信息面板
        panel_height = 140
        cv.rectangle(result_frame, (10, 10), (450, panel_height), (30, 30, 30), -1)
        cv.rectangle(result_frame, (10, 10), (450, panel_height), (255, 255, 255), 2)
        
        # 文本信息
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        # 帧号
        cv.putText(result_frame, f"Frame: {frame_number}", (20, 35), 
                   font, font_scale, (255, 255, 255), thickness)
        
        # 模板名称
        template_name = os.path.splitext(os.path.basename(self.template_path))[0]
        cv.putText(result_frame, f"Template: {template_name}", (20, 60), 
                   font, font_scale, (255, 255, 255), thickness)
        
        # 关键点检测状态
        landmark_color = (0, 255, 0) if has_landmarks else (0, 0, 255)
        landmark_status = "DETECTED" if has_landmarks else "NOT DETECTED"
        cv.putText(result_frame, f"Landmarks: {landmark_status}", (20, 85), 
                   font, font_scale, landmark_color, thickness)
        
        if has_landmarks:
            # 相似度 - 根据数值使用不同颜色
            sim_color = (0, 255, 0) if similarity > self.similarity_threshold else (0, 165, 255)
            cv.putText(result_frame, f"Similarity: {similarity:.2f}", (20, 110), 
                       font, font_scale, sim_color, thickness)
            
            # 匹配状态
            status_color = (0, 255, 0) if matched else (0, 0, 255)
            status_text = "MATCHED!" if matched else "NO MATCH"
            cv.putText(result_frame, f"Status: {status_text}", (20, 135), 
                       font, font_scale, status_color, thickness)
            
            # 在右侧绘制相似度条形图
            bar_x = 470
            bar_y = 20
            bar_width = 20
            bar_height = 100
            
            # 绘制背景条
            cv.rectangle(result_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
            cv.rectangle(result_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
            
            # 绘制相似度条
            fill_height = int((similarity / 100.0) * bar_height)
            if fill_height > 0:
                fill_color = (0, 255, 0) if similarity > self.similarity_threshold else (0, 165, 255)
                cv.rectangle(result_frame, 
                            (bar_x + 2, bar_y + bar_height - fill_height), 
                            (bar_x + bar_width - 2, bar_y + bar_height - 2), 
                            fill_color, -1)
            
            # 绘制阈值线
            threshold_y = bar_y + bar_height - int((self.similarity_threshold / 100.0) * bar_height)
            cv.line(result_frame, (bar_x - 5, threshold_y), (bar_x + bar_width + 5, threshold_y), (255, 255, 0), 2)
            
            # 添加阈值标签
            cv.putText(result_frame, f"{self.similarity_threshold:.0f}", (bar_x + bar_width + 10, threshold_y + 5), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        else:
            # 无关键点时显示N/A
            cv.putText(result_frame, f"Similarity: N/A", (20, 110), 
                       font, font_scale, (128, 128, 128), thickness)
            cv.putText(result_frame, f"Status: NO HAND", (20, 135), 
                       font, font_scale, (0, 0, 255), thickness)
        
        return result_frame
    
    def _visualize_no_detection(self, frame, frame_number):
        """可视化处理失败的帧"""
        result_frame = frame.copy()
        
        # 绘制信息面板
        panel_height = 120
        cv.rectangle(result_frame, (10, 10), (400, panel_height), (30, 30, 30), -1)
        cv.rectangle(result_frame, (10, 10), (400, panel_height), (255, 255, 255), 2)
        
        # 文本信息
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        # 帧号
        cv.putText(result_frame, f"Frame: {frame_number}", (20, 35), 
                   font, font_scale, (255, 255, 255), thickness)
        
        # 错误状态
        cv.putText(result_frame, f"Status: PROCESSING ERROR", (20, 60), 
                   font, font_scale, (0, 0, 255), thickness)
        
        cv.putText(result_frame, f"Similarity: N/A", (20, 85), 
                   font, font_scale, (128, 128, 128), thickness)
        
        return result_frame
    
    def _print_statistics(self, processed_count, landmarks_detected_count, match_count, similarities):
        """打印统计结果"""
        print("-" * 80)
        print(f"\n测试完成!")
        print(f"处理帧数: {processed_count}")
        print(f"检测到手部关键点: {landmarks_detected_count}")
        print(f"匹配成功: {match_count}")
        
        if landmarks_detected_count > 0:
            detection_rate = (landmarks_detected_count / processed_count) * 100
            print(f"关键点检测率: {detection_rate:.1f}%")
        
        if similarities:
            match_rate = (match_count / len(similarities)) * 100
            print(f"匹配成功率: {match_rate:.1f}%")
            
            print(f"\n相似度统计:")
            print(f"  最高: {max(similarities):.2f}")
            print(f"  最低: {min(similarities):.2f}")
            print(f"  平均: {np.mean(similarities):.2f}")
            print(f"  中位数: {np.median(similarities):.2f}")
            print(f"  标准差: {np.std(similarities):.2f}")
            
            # 相似度区间分布
            print(f"\n相似度分布:")
            ranges = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
            for low, high in ranges:
                count = sum(1 for s in similarities if low <= s < high)
                percentage = count / len(similarities) * 100
                print(f"  {low:2d}-{high:2d}: {count:3d} 帧 ({percentage:5.1f}%)")
        
        else:
            print("未检测到任何手部关键点进行相似度计算")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='基于MediaPipe的手势匹配测试工具')
    parser.add_argument('video_path', help='视频文件路径')
    parser.add_argument('--template', type=str, 
                        default='templates/group1/weasel.jpg',
                        help='模板图片路径')
    parser.add_argument('--skip', type=int, default=3,
                        help='帧跳跃间隔，默认为3')
    parser.add_argument('--threshold', type=float, default=75,
                        help='相似度阈值，默认为75 (MediaPipe通常需要更高阈值)')
    parser.add_argument('--max-frames', type=int,
                        help='最大处理帧数')
    parser.add_argument('--save-results', action='store_true',
                        help='保存结果图像到test_results目录')
    parser.add_argument('--no-video', action='store_true',
                        help='不生成输出视频文件')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.video_path):
        print(f"错误: 视频文件不存在: {args.video_path}")
        return
    
    if not os.path.exists(args.template):
        print(f"错误: 模板文件不存在: {args.template}")
        return
    
    try:
        # 创建测试器
        tester = HandGestureTester(args.template, args.threshold)
        
        # 运行测试
        output_video = not args.no_video
        tester.test_video(args.video_path, args.skip, args.max_frames, args.save_results, output_video)
    
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()