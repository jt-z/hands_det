"""
基于Inception V3的手势匹配测试脚本
从视频中抽取帧并与模板图片进行相似度比较
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
    from handpose_shadow.inception_image_matcher import InceptionImageMatcher
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保在正确的项目目录下运行此脚本")
    print("需要安装 torch, torchvision, scikit-learn, Pillow")
    sys.exit(1)

class HandGestureTester:
    """基于Inception V3的手势匹配测试器"""
    
    def __init__(self, template_path, model_path='HSPR_InceptionV3.pt', similarity_threshold=75):
        """
        初始化测试器
        
        参数:
            template_path (str): 模板图片路径
            model_path (str): Inception V3模型权重路径
            similarity_threshold (float): 相似度阈值
        """
        self.template_path = template_path
        self.similarity_threshold = similarity_threshold
        
        # 初始化InceptionImageMatcher
        self.image_matcher = InceptionImageMatcher(
            model_path=model_path,
            default_threshold=similarity_threshold
        )
        
        # 加载模板图像
        self.template_image = self._load_template_image()
        if self.template_image is None:
            raise ValueError(f"无法加载模板图像: {template_path}")
        
        print(f"模板图像加载成功: {template_path}")
        print(f"模板图像尺寸: {self.template_image.shape}")
        print(f"相似度阈值: {similarity_threshold}")
        print(f"使用设备: {self.image_matcher.device}")
    
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
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_video_path = f"output_{video_name}_inception_test.mp4"
            
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            video_writer = cv.VideoWriter(output_video_path, fourcc, max(1, fps/frame_skip), (width, height))
            
            if video_writer.isOpened():
                print(f"  输出视频: {output_video_path}")
            else:
                print("警告: 无法创建输出视频")
                video_writer = None
                output_video = False
        
        # 统计变量
        frame_count = 0
        processed_count = 0
        match_count = 0
        similarities = []
        
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
                similarity = self._process_frame(frame, frame_count)
                
                if similarity > 0:  # 成功处理
                    similarities.append(similarity)
                    matched = similarity > self.similarity_threshold
                    
                    if matched:
                        match_count += 1
                    
                    # 打印结果
                    status = "✓ 匹配" if matched else "✗ 不匹配"
                    print(f"帧 {frame_count:6d}: 相似度 {similarity:6.2f} - {status}")
                    
                    # 创建可视化帧
                    result_frame = self._visualize_result(frame, similarity, matched, frame_count)
                    
                    # 保存结果图像
                    if save_results:
                        output_path = os.path.join(results_dir, f"frame_{frame_count:06d}_sim_{similarity:.1f}.jpg")
                        cv.imwrite(output_path, result_frame)
                
                else:
                    print(f"帧 {frame_count:6d}: 处理失败")
                    result_frame = self._visualize_error(frame, frame_count)
                
                # 写入视频帧
                if video_writer is not None:
                    if result_frame.shape[:2] != (height, width):
                        result_frame = cv.resize(result_frame, (width, height))
                    video_writer.write(result_frame)
        
        except KeyboardInterrupt:
            print("\n用户中断处理")
        
        finally:
            cap.release()
            if video_writer is not None:
                video_writer.release()
                if output_video_path and os.path.exists(output_video_path):
                    file_size = os.path.getsize(output_video_path) / (1024 * 1024)
                    print(f"\n✓ 结果视频已保存: {output_video_path} ({file_size:.1f} MB)")
        
        # 输出统计结果
        self._print_statistics(processed_count, match_count, similarities)
    
    def _process_frame(self, frame, frame_number):
        """处理单帧，返回相似度分数"""
        try:
            # 直接调用 _compare_images 方法
            similarity = self.image_matcher._compare_images(self.template_image, frame)
            
            if frame_number % 30 == 0:  # 每30帧打印一次调试信息
                print(f"  调试 - 帧 {frame_number}: 相似度 {similarity:.2f}")
            
            return similarity
            
        except Exception as e:
            print(f"处理帧 {frame_number} 时出错: {e}")
            return 0.0
    
    def _visualize_result(self, frame, similarity, matched, frame_number):
        """可视化检测结果"""
        result_frame = frame.copy()
        h_frame, w_frame = frame.shape[:2]
        
        # 绘制模板缩略图（右上角）
        template_size = min(w_frame, h_frame) // 4
        template_resized = cv.resize(self.template_image, (template_size, template_size))
        result_frame[10:10+template_size, w_frame-template_size-10:w_frame-10] = template_resized
        
        # 绘制模板边框
        cv.rectangle(result_frame, 
                    (w_frame-template_size-10, 10), 
                    (w_frame-10, 10+template_size), 
                    (255, 255, 255), 2)
        
        # 绘制信息面板
        panel_height = 120
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
        
        # 相似度
        sim_color = (0, 255, 0) if similarity > self.similarity_threshold else (0, 165, 255)
        cv.putText(result_frame, f"Similarity: {similarity:.2f}", (20, 85), 
                   font, font_scale, sim_color, thickness)
        
        # 匹配状态
        status_color = (0, 255, 0) if matched else (0, 0, 255)
        status_text = "MATCHED!" if matched else "NO MATCH"
        cv.putText(result_frame, f"Status: {status_text}", (20, 110), 
                   font, font_scale, status_color, thickness)
        
        # 相似度条形图
        bar_x = 470
        bar_y = 20
        bar_width = 20
        bar_height = 80
        
        # 背景条
        cv.rectangle(result_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
        cv.rectangle(result_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
        
        # 相似度条
        fill_height = int((similarity / 100.0) * bar_height)
        if fill_height > 0:
            fill_color = (0, 255, 0) if similarity > self.similarity_threshold else (0, 165, 255)
            cv.rectangle(result_frame, 
                        (bar_x + 2, bar_y + bar_height - fill_height), 
                        (bar_x + bar_width - 2, bar_y + bar_height - 2), 
                        fill_color, -1)
        
        # 阈值线
        threshold_y = bar_y + bar_height - int((self.similarity_threshold / 100.0) * bar_height)
        cv.line(result_frame, (bar_x - 5, threshold_y), (bar_x + bar_width + 5, threshold_y), (255, 255, 0), 2)
        
        return result_frame
    
    def _visualize_error(self, frame, frame_number):
        """可视化处理失败的帧"""
        result_frame = frame.copy()
        
        # 绘制信息面板
        panel_height = 100
        cv.rectangle(result_frame, (10, 10), (400, panel_height), (30, 30, 30), -1)
        cv.rectangle(result_frame, (10, 10), (400, panel_height), (255, 255, 255), 2)
        
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        cv.putText(result_frame, f"Frame: {frame_number}", (20, 35), 
                   font, font_scale, (255, 255, 255), thickness)
        cv.putText(result_frame, f"Status: PROCESSING ERROR", (20, 60), 
                   font, font_scale, (0, 0, 255), thickness)
        cv.putText(result_frame, f"Similarity: N/A", (20, 85), 
                   font, font_scale, (128, 128, 128), thickness)
        
        return result_frame
    
    def _print_statistics(self, processed_count, match_count, similarities):
        """打印统计结果"""
        print("-" * 80)
        print(f"\n测试完成!")
        print(f"处理帧数: {processed_count}")
        print(f"匹配成功: {match_count}")
        
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
            print("未成功处理任何帧")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='基于Inception V3的手势匹配测试工具')
    parser.add_argument('video_path', help='视频文件路径')
    parser.add_argument('--template', type=str, 
                        default='templates/group1/weasel.jpg',
                        help='模板图片路径')
    parser.add_argument('--model', type=str, 
                        default='HSPR_InceptionV3.pt',
                        help='Inception V3模型权重路径')
    parser.add_argument('--skip', type=int, default=3,
                        help='帧跳跃间隔，默认为3')
    parser.add_argument('--threshold', type=float, default=75,
                        help='相似度阈值，默认为75')
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
    
    if not os.path.exists(args.model):
        print(f"错误: 模型权重文件不存在: {args.model}")
        return
    
    try:
        # 创建测试器
        tester = HandGestureTester(args.template, args.model, args.threshold)
        
        # 运行测试
        output_video = not args.no_video
        tester.test_video(args.video_path, args.skip, args.max_frames, args.save_results, output_video)
    
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


# 测试1：
# .\handpose_shadow\inception_image_matcher_test.py .\core_methods\inception_vectors\record_video_small.mp4 --template   .\handpose_shadow\templates\group1\hedgehog.jpg --model .\core_methods\inception_vectors\HSPR_InceptionV3.pt --threshold 75

# 测试2：
