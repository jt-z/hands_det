"""
手部检测模块
负责从视频帧中检测手部并提取轮廓
"""

import cv2 as cv
import numpy as np
from .config import SKIN_LOWER_HSV, SKIN_UPPER_HSV, MIN_CONTOUR_AREA, FRAME_WIDTH, FRAME_HEIGHT
from .utils.logging_utils import get_logger
from .utils import resize_image

class HandDetector:
    """手部检测器类，用于检测和提取手部轮廓"""
    
    def __init__(self, skin_lower=None, skin_upper=None, min_area=None, resize_frame=True):
        """
        初始化手部检测器
        
        参数:
            skin_lower (list, 可选): HSV空间皮肤颜色下限，默认使用配置值
            skin_upper (list, 可选): HSV空间皮肤颜色上限，默认使用配置值
            min_area (int, 可选): 最小手部轮廓面积，默认使用配置值
            resize_frame (bool): 是否调整输入帧大小
        """
        self.logger = get_logger("hand_detector")
        
        # 使用传入的参数或默认配置
        self.skin_lower = np.array(skin_lower or SKIN_LOWER_HSV, dtype=np.uint8)
        self.skin_upper = np.array(skin_upper or SKIN_UPPER_HSV, dtype=np.uint8)
        self.min_area = min_area or MIN_CONTOUR_AREA
        self.resize_frame = resize_frame
        
        self.logger.info(f"HandDetector initialized with: skin_range=[{self.skin_lower}, {self.skin_upper}], "
                        f"min_area={self.min_area}")
    
    def detect_hand(self, frame):
        """
        检测手部并返回掩码和轮廓
        
        参数:
            frame (numpy.ndarray): 输入帧
            
        返回:
            tuple: (掩码, 最大轮廓)，如果未检测到轮廓则轮廓为None
        """
        if frame is None:
            self.logger.warning("Received None frame")
            return None, None
        
        # 如果需要，调整帧大小
        if self.resize_frame and (frame.shape[1] != FRAME_WIDTH or frame.shape[0] != FRAME_HEIGHT):
            frame = resize_image(frame, FRAME_WIDTH, FRAME_HEIGHT)
        
        # 生成皮肤掩码
        skin_mask = self._detect_skin(frame)
        
        # 提取轮廓
        contours = self._extract_contours(skin_mask)
        
        # 选择最大的轮廓
        main_contour = self._select_main_contour(contours)
        
        if main_contour is not None:
            self.logger.debug(f"Detected hand contour with area: {cv.contourArea(main_contour)}")
        else:
            self.logger.debug("No hand contour detected")
        
        return skin_mask, main_contour
    
    def _detect_skin(self, frame):
        """
        检测皮肤区域
        
        参数:
            frame (numpy.ndarray): 输入帧
            
        返回:
            numpy.ndarray: 皮肤掩码
        """
        # 转换到HSV颜色空间
        hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        
        # 创建皮肤掩码
        skin_mask = cv.inRange(hsv_frame, self.skin_lower, self.skin_upper)
        
        # 应用形态学操作来改善掩码
        kernel = np.ones((3, 3), np.uint8)
        skin_mask = cv.erode(skin_mask, kernel, iterations=1)
        skin_mask = cv.dilate(skin_mask, kernel, iterations=2)
        
        # 应用高斯模糊
        skin_mask = cv.GaussianBlur(skin_mask, (5, 5), 0)
        
        return skin_mask
    
    def _extract_contours(self, mask):
        """
        从掩码中提取轮廓
        
        参数:
            mask (numpy.ndarray): 二值掩码
            
        返回:
            list: 检测到的轮廓列表
        """
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        return contours
    
    def _select_main_contour(self, contours):
        """
        选择主要手部轮廓
        
        参数:
            contours (list): 轮廓列表
            
        返回:
            numpy.ndarray: 主要轮廓，如果没有则为None
        """
        if not contours:
            return None
        
        # 过滤面积小于阈值的轮廓
        valid_contours = [cnt for cnt in contours if cv.contourArea(cnt) > self.min_area]
        
        if not valid_contours:
            return None
        
        # 返回面积最大的轮廓
        return max(valid_contours, key=cv.contourArea)
    
    def draw_detection(self, frame, mask=None, contour=None, show_mask=True):
        """
        在帧上绘制检测结果
        
        参数:
            frame (numpy.ndarray): 输入帧
            mask (numpy.ndarray, 可选): 皮肤掩码
            contour (numpy.ndarray, 可选): 手部轮廓
            show_mask (bool): 是否显示掩码
            
        返回:
            numpy.ndarray: 带有标记的帧
        """
        result = frame.copy()
        
        # 如果提供了轮廓，绘制轮廓
        if contour is not None:
            cv.drawContours(result, [contour], -1, (0, 255, 0), 2)
            
            # 绘制轮廓的外接矩形
            x, y, w, h = cv.boundingRect(contour)
            cv.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # 添加轮廓面积信息
            area = cv.contourArea(contour)
            cv.putText(result, f"Area: {int(area)}", (x, y-10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 如果提供了掩码且需要显示，将掩码叠加到结果帧上
        if mask is not None and show_mask:
            # 创建彩色掩码用于叠加
            color_mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
            # 掩码区域设为半透明红色
            red_mask = np.zeros_like(color_mask)
            red_mask[:, :] = (0, 0, 200)  # BGR格式，红色
            mask_overlay = cv.bitwise_and(red_mask, red_mask, mask=mask)
            # 将掩码叠加到结果上，使用0.3的透明度
            cv.addWeighted(result, 1.0, mask_overlay, 0.3, 0, result)
        
        return result