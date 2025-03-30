"""
工具模块初始化文件
"""

import cv2 as cv
import numpy as np
import os
import time
from datetime import datetime

def resize_image(image, width=None, height=None):
    """
    调整图像大小，保持宽高比
    
    参数:
        image: 输入图像
        width: 目标宽度 (可选)
        height: 目标高度 (可选)
    
    返回:
        调整大小后的图像
    """
    if width is None and height is None:
        return image
        
    h, w = image.shape[:2]
    
    if width is None:
        aspect_ratio = height / float(h)
        new_width = int(w * aspect_ratio)
        return cv.resize(image, (new_width, height))
    
    elif height is None:
        aspect_ratio = width / float(w)
        new_height = int(h * aspect_ratio)
        return cv.resize(image, (width, new_height))
    
    else:
        return cv.resize(image, (width, height))

def ensure_dir(directory):
    """
    确保目录存在，如果不存在则创建
    
    参数:
        directory: 目录路径
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def current_timestamp():
    """
    获取当前时间戳字符串，格式: YYYY-MM-DD HH:MM:SS.fff
    
    返回:
        时间戳字符串
    """
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

def draw_contour_info(image, contour, text, position=None, color=(0, 255, 0), thickness=2):
    """
    在图像上绘制轮廓和信息文本
    
    参数:
        image: 输入图像
        contour: 要绘制的轮廓
        text: 要显示的文本
        position: 文本位置，如果为None则在轮廓上方显示
        color: 颜色 (B,G,R)
        thickness: 线条粗细
    
    返回:
        带有轮廓和文本的图像
    """
    result = image.copy()
    
    # 绘制轮廓
    cv.drawContours(result, [contour], -1, color, thickness)
    
    # 确定文本位置
    if position is None:
        x, y, w, h = cv.boundingRect(contour)
        position = (x, y - 10)
    
    # 绘制文本
    cv.putText(
        result, 
        text, 
        position, 
        cv.FONT_HERSHEY_SIMPLEX, 
        0.6, 
        color, 
        thickness=2
    )
    
    return result

def measure_fps(func):
    """
    测量函数执行帧率的装饰器
    
    使用方法:
    @measure_fps
    def process_frame(frame):
        pass
    """
    last_time = time.time()
    fps_counter = 0
    fps = 0
    
    def wrapped(*args, **kwargs):
        nonlocal last_time, fps_counter, fps
        
        # 计算FPS
        current_time = time.time()
        fps_counter += 1
        
        if current_time - last_time > 1.0:  # 每秒更新一次
            fps = fps_counter
            fps_counter = 0
            last_time = current_time
        
        # 调用原函数
        result = func(*args, **kwargs)
        
        # 如果结果是图像，则在图像上显示FPS
        if isinstance(result, np.ndarray) and len(result.shape) >= 2:
            cv.putText(
                result, 
                f"FPS: {fps}", 
                (10, 30), 
                cv.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0), 
                2
            )
        
        return result
    
    return wrapped