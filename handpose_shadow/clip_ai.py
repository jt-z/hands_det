import cv2
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F
import os
from pathlib import Path

class VideoShadowMatcher:
    def __init__(self, shadow_images_dir="group1_demo_images"):
        """
        初始化视频手影匹配器
        """
        print("正在加载 CLIP 模型...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.to(self.device)
        self.model.eval()
        
        # 加载动物手影图片
        self.shadow_images = {}
        self.shadow_embeddings = {}
        self.load_shadow_images(shadow_images_dir)
        
        print(f"模型加载完成，使用设备: {self.device}")
        print(f"已加载 {len(self.shadow_images)} 张动物手影图片")
    
    def load_shadow_images(self, images_dir):
        """加载并预计算动物手影图片的嵌入"""
        image_dir = Path(images_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"图片目录不存在: {images_dir}")
        
        # 支持的图片格式
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        for img_path in image_dir.iterdir():
            if img_path.suffix.lower() in valid_extensions:
                try:
                    # 加载图片
                    image = Image.open(img_path).convert('RGB')
                    self.shadow_images[img_path.stem] = image
                    
                    # 预计算嵌入
                    embedding = self.get_image_embedding(image)
                    self.shadow_embeddings[img_path.stem] = embedding
                    
                    print(f"已加载: {img_path.name}")
                except Exception as e:
                    print(f"加载图片失败 {img_path.name}: {e}")
    
    def get_image_embedding(self, image):
        """获取图像的CLIP嵌入"""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        return F.normalize(image_features, p=2, dim=1)
    
    def calculate_similarities(self, frame_embedding):
        """计算帧与所有动物手影的相似度"""
        similarities = {}
        for name, shadow_embedding in self.shadow_embeddings.items():
            similarity = torch.cosine_similarity(frame_embedding, shadow_embedding)
            similarities[name] = similarity.item()
        return similarities
    
    def draw_similarity_info(self, frame, similarities, frame_count, fps):
        """在帧上绘制相似度信息"""
        # 创建副本避免修改原帧
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]
        
        # 排序相似度（从高到低）
        sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        # 绘制背景框
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (10, 10), (min(400, w-10), min(200, h-10)), (0, 0, 0), -1)
        display_frame = cv2.addWeighted(display_frame, 0.7, overlay, 0.3, 0)
        
        # 绘制标题
        cv2.putText(display_frame, "Animal Shadow Similarity", 
                   (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 绘制时间信息
        time_text = f"Frame: {frame_count} | Time: {frame_count/fps:.1f}s"
        cv2.putText(display_frame, time_text, 
                   (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # 绘制相似度排名
        y_offset = 80
        for i, (name, similarity) in enumerate(sorted_similarities):
            # 格式化动物名称（移除文件扩展名，美化显示）
            display_name = name.replace('_', ' ').title()
            
            # 相似度百分比
            similarity_percent = similarity * 100
            
            # 根据相似度设置颜色
            if similarity > 0.8:
                color = (0, 255, 0)  # 绿色 - 高相似度
            elif similarity > 0.6:
                color = (0, 255, 255)  # 黄色 - 中等相似度
            else:
                color = (255, 255, 255)  # 白色 - 低相似度
            
            # 如果是最高相似度，特别标注
            prefix = "★ " if i == 0 else f"{i+1}. "
            
            text = f"{prefix}{display_name}: {similarity_percent:.1f}%"
            cv2.putText(display_frame, text, 
                       (20, y_offset + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 绘制最佳匹配的进度条
        best_name, best_similarity = sorted_similarities[0]
        bar_width = int(300 * max(0, best_similarity))
        cv2.rectangle(display_frame, (20, y_offset + len(sorted_similarities) * 25 + 10), 
                     (20 + bar_width, y_offset + len(sorted_similarities) * 25 + 25), 
                     (0, 255, 0), -1)
        cv2.rectangle(display_frame, (20, y_offset + len(sorted_similarities) * 25 + 10), 
                     (320, y_offset + len(sorted_similarities) * 25 + 25), 
                     (255, 255, 255), 1)
        
        return display_frame
    
    def process_video(self, video_path, output_path=None, skip_frames=3):
        """
        处理视频，逐帧比较相似度
        
        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径（可选）
            skip_frames: 跳帧数量，用于加速处理
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        # 获取视频信息
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"视频信息: {width}x{height}, {fps}FPS, {total_frames}帧")
        
        # 设置输出视频
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        if output_path:
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        processed_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 跳帧处理以提高速度
                if frame_count % (skip_frames + 1) == 0:
                    # 转换为PIL图像进行CLIP处理
                    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    
                    # 获取帧嵌入
                    frame_embedding = self.get_image_embedding(pil_frame)
                    
                    # 计算相似度
                    similarities = self.calculate_similarities(frame_embedding)
                    
                    # 绘制相似度信息
                    display_frame = self.draw_similarity_info(frame, similarities, frame_count, fps)
                    
                    # 显示结果
                    # cv2.imshow('Video Shadow Similarity', display_frame)
                    
                    # 保存到输出视频
                    if output_path:
                        out.write(display_frame)
                    
                    processed_count += 1
                    
                    # 打印进度
                    if processed_count % 30 == 0:
                        progress = (frame_count / total_frames) * 100
                        best_match = max(similarities.items(), key=lambda x: x[1])
                        print(f"处理进度: {progress:.1f}% | 当前最佳匹配: {best_match[0]} ({best_match[1]:.3f})")
                
                frame_count += 1
                
                # 按 'q' 键退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("用户中断处理")
                    break
        
        except KeyboardInterrupt:
            print("处理被中断")
        
        finally:
            cap.release()
            if output_path:
                out.release()
            cv2.destroyAllWindows()
            
            print(f"处理完成! 共处理 {processed_count} 帧")
            if output_path:
                print(f"输出视频已保存: {output_path}")

def main():
    """主函数"""
    try:
        # 创建匹配器实例
        matcher = VideoShadowMatcher("group1_demo_images")
        
        # 处理视频
        video_path = "../assets/self_get.mov"
        output_path = "demo_with_similarity_self_get.mp4"  # 可选：保存带相似度信息的视频
        
        # 开始处理（skip_frames=2 表示每3帧处理一次，可根据需要调整）
        matcher.process_video(video_path, output_path, skip_frames=2)
        
    except Exception as e:
        print(f"处理出错: {e}")
        print("请确保:")
        print("1. demo.mp4 视频文件存在")
        print("2. group1_demo_images 目录存在且包含图片")
        print("3. 已安装所需依赖: pip install torch transformers opencv-python pillow")

if __name__ == "__main__":
    main()
