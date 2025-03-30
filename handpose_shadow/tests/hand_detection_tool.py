"""
手部检测工具 (新版本)
用于测试HandDetector类处理图片或视频
"""

import os
import sys
import cv2 as cv
import numpy as np
import argparse


import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from handpose_shadow.hand_detector import HandDetector
from handpose_shadow.utils import resize_image

def process_image(image_path, output_path=None):
    """处理单张图片"""
    # 读取图片
    image = cv.imread(image_path)
    if image is None:
        print(f"无法读取图片: {image_path}")
        return False
    
    # 创建手部检测器
    detector = HandDetector(min_area=1000)
    
    # 检测手部
    mask, contour = detector.detect_hand(image)
    
    # 绘制检测结果
    if contour is not None:
        # 自己绘制结果，避免使用detector.draw_detection中有问题的部分
        result = image.copy()
        
        # 绘制轮廓
        cv.drawContours(result, [contour], -1, (0, 255, 0), 2)
        
        # 绘制轮廓的外接矩形
        x, y, w, h = cv.boundingRect(contour)
        cv.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # 添加轮廓面积信息
        area = cv.contourArea(contour)
        cv.putText(result, f"Area: {int(area)}", (x, y-10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        print(f"检测到手部轮廓，面积: {area}")
        
        # 显示结果
        cv.imshow("检测结果", result)
        cv.waitKey(0)
        cv.destroyAllWindows()
        
        # 如果指定了输出路径，则保存结果
        if output_path:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            # 如果未指定扩展名，添加.png
            if not os.path.splitext(output_path)[1]:
                output_path += '.png'
                
            cv.imwrite(output_path, result)
            print(f"结果已保存至: {output_path}")
        
        return True
    else:
        print("未检测到手部")
        return False

def process_video(video_path, output_path=None):
    """处理视频"""
    # 打开视频
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return False
    
    # 获取视频信息
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv.CAP_PROP_FPS)
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    
    print(f"视频分辨率: {width}x{height}, FPS: {fps}, 总帧数: {total_frames}")
    
    # 创建手部检测器
    detector = HandDetector(min_area=1000)
    
    # 设置输出视频
    out = None
    if output_path:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 如果未指定扩展名，添加.avi
        if not os.path.splitext(output_path)[1]:
            output_path += '.avi'
            
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        out = cv.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 处理视频帧
    frame_count = 0
    skip_frames = 2  # 每隔几帧处理一次，减轻计算负担
    
    while cap.isOpened():
        ret, frame = cap.read()
        frame_count += 1
        
        if not ret:
            break
        
        # 跳帧处理
        if frame_count % skip_frames != 0:
            if out:
                out.write(frame)
            continue
        
        # 检测手部
        mask, contour = detector.detect_hand(frame)
        
        # 绘制检测结果
        if contour is not None:
            # 自己绘制结果，避免使用detector.draw_detection中有问题的部分
            result = frame.copy()
            
            # 绘制轮廓
            cv.drawContours(result, [contour], -1, (0, 255, 0), 2)
            
            # 绘制轮廓的外接矩形
            x, y, w, h = cv.boundingRect(contour)
            cv.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # 在帧上显示计数和面积信息
            cv.putText(
                result,
                f"Frame: {frame_count}/{total_frames} | Area: {int(cv.contourArea(contour))}",
                (10, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            # 显示结果
            cv.imshow("检测结果", result)
            
            # 如果设置了输出，保存帧
            if out:
                out.write(result)
        else:
            # 未检测到手部，直接写入原始帧
            if out:
                out.write(frame)
            
            # 显示原始帧
            cv.putText(
                frame,
                f"Frame: {frame_count}/{total_frames} (无手部)",
                (10, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
            cv.imshow("检测结果", frame)
        
        # 每5%的处理进度打印一次
        if total_frames > 0 and frame_count % (total_frames // 20 + 1) == 0:
            progress = frame_count / total_frames * 100
            print(f"处理进度: {progress:.1f}%")
        
        # 按ESC键退出
        if cv.waitKey(1) & 0xFF == 27:
            break
    
    # 释放资源
    cap.release()
    if out:
        out.release()
    cv.destroyAllWindows()
    
    print("视频处理完成")
    return True

def process_camera(camera_id=0, output_path=None):
    """使用摄像头进行实时检测"""
    # 打开摄像头
    cap = cv.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"无法打开摄像头 ID: {camera_id}")
        return False
    
    # 获取摄像头信息
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv.CAP_PROP_FPS)
    
    print(f"摄像头分辨率: {width}x{height}, FPS: {fps}")
    
    # 创建手部检测器
    detector = HandDetector(min_area=1000)
    
    # 设置输出视频
    out = None
    if output_path:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 如果未指定扩展名，添加.avi
        if not os.path.splitext(output_path)[1]:
            output_path += '.avi'
            
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        out = cv.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 处理视频帧
    frame_count = 0
    
    print("按ESC键退出")
    while cap.isOpened():
        ret, frame = cap.read()
        frame_count += 1
        
        if not ret:
            break
        
        # 检测手部
        mask, contour = detector.detect_hand(frame)
        
        # 绘制检测结果
        if contour is not None:
            # 自己绘制结果，避免使用detector.draw_detection中有问题的部分
            result = frame.copy()
            
            # 绘制轮廓
            cv.drawContours(result, [contour], -1, (0, 255, 0), 2)
            
            # 绘制轮廓的外接矩形
            x, y, w, h = cv.boundingRect(contour)
            cv.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # 在帧上显示信息
            cv.putText(
                result,
                f"手部面积: {cv.contourArea(contour):.0f}",
                (10, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            # 显示结果
            cv.imshow("手部检测", result)
            
            # 如果设置了输出，保存帧
            if out:
                out.write(result)
        else:
            # 未检测到手部，直接写入原始帧
            cv.putText(
                frame,
                "未检测到手部",
                (10, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
            cv.imshow("手部检测", frame)
            
            if out:
                out.write(frame)
        
        # 按ESC键退出
        if cv.waitKey(1) & 0xFF == 27:
            break
    
    # 释放资源
    cap.release()
    if out:
        out.release()
    cv.destroyAllWindows()
    
    print("摄像头处理结束")
    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="手部检测测试工具")
    
    # 设置输入源
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--image", type=str, help="输入图片路径")
    source_group.add_argument("--video", type=str, help="输入视频路径")
    source_group.add_argument("--camera", type=int, default=None, nargs='?', const=0, help="使用摄像头（默认ID: 0）")
    
    # 输出设置
    parser.add_argument("--output", type=str, help="输出文件路径")
    
    # 调整检测参数
    parser.add_argument("--min-area", type=int, default=1000, help="最小手部轮廓面积")
    parser.add_argument("--skin-hue-low", type=int, default=0, help="HSV色调下限")
    parser.add_argument("--skin-hue-high", type=int, default=20, help="HSV色调上限")
    parser.add_argument("--skin-sat-low", type=int, default=48, help="HSV饱和度下限")
    parser.add_argument("--skin-sat-high", type=int, default=255, help="HSV饱和度上限")
    parser.add_argument("--skin-val-low", type=int, default=80, help="HSV亮度下限")
    parser.add_argument("--skin-val-high", type=int, default=255, help="HSV亮度上限")
    
    args = parser.parse_args()
    
    print("手部检测工具 (新版本)")
    
    # 根据输入类型调用相应的处理函数
    if args.image:
        print(f"处理图片: {args.image}")
        process_image(args.image, args.output)
    elif args.video:
        print(f"处理视频: {args.video}")
        process_video(args.video, args.output)
    elif args.camera is not None:
        print(f"使用摄像头ID: {args.camera}")
        process_camera(args.camera, args.output)

if __name__ == "__main__":
    main()