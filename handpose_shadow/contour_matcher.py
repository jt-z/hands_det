"""
轮廓匹配模块
负责比较手部轮廓与模板轮廓，计算相似度
"""

import cv2 as cv
import numpy as np
from .config import SIMILARITY_THRESHOLD
from .utils.logging_utils import get_logger, LogPerformance

class ContourMatcher:
    """轮廓匹配器类，用于比较手部轮廓与模板轮廓"""
    
    def __init__(self, default_threshold=None):
        """
        初始化轮廓匹配器
        
        参数:
            default_threshold (float, 可选): 默认相似度阈值，默认使用配置值
        """
        self.logger = get_logger("contour_matcher")
        
        # 使用传入的阈值或默认配置
        self.default_threshold = default_threshold or SIMILARITY_THRESHOLD
        
        self.logger.info(f"ContourMatcher initialized with default_threshold={self.default_threshold}")
    
    @LogPerformance()
    def match_with_templates(self, hand_contour, templates):
        """
        将手部轮廓与多个模板比较，返回最佳匹配
        
        参数:
            hand_contour (numpy.ndarray): 手部轮廓
            templates (dict): 模板字典，格式为 {template_id: template_data, ...}
                template_data 应包含 'contour', 'name', 'threshold' 等键
            
        返回:
            dict: 匹配结果，包含匹配信息和相似度得分
                 如果没有找到匹配或 hand_contour 为 None，则返回 None
        """
        if hand_contour is None:
            self.logger.warning("Cannot match: hand_contour is None")
            return None
        
        if not templates:
            self.logger.warning("Cannot match: no templates provided")
            return None
        
        results = []
        
        # 比较手部轮廓与每个模板
        for template_id, template in templates.items():
            template_contour = template.get("contour")
            if template_contour is None:
                self.logger.warning(f"Template {template_id} has no contour")
                continue
            
            # 计算相似度
            similarity = self._compare_contours(template_contour, hand_contour)
            
            # 获取模板特定阈值或使用默认值
            threshold = template.get("threshold", self.default_threshold)
            
            # 记录结果
            results.append({
                "id": template_id,
                "name": template.get("name", template_id),
                "similarity": similarity,
                "threshold": threshold,
                "matched": similarity > threshold
            })
        
        # 按相似度排序
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        if not results:
            return None
        
        # 记录最佳匹配
        best_match = results[0]
        self.logger.debug(f"Best match: {best_match['name']} "
                         f"(similarity: {best_match['similarity']:.2f}, "
                         f"threshold: {best_match['threshold']}, "
                         f"matched: {best_match['matched']})")
        
        return best_match
    
    def _compare_contours(self, contour1, contour2):
        """
        比较两个轮廓的相似度
        
        参数:
            contour1 (numpy.ndarray): 第一个轮廓
            contour2 (numpy.ndarray): 第二个轮廓
            
        返回:
            float: 相似度得分 (0-100)，越高表示越相似
        """
        try:
            # 使用OpenCV的轮廓匹配函数
            ret = cv.matchShapes(contour1, contour2, cv.CONTOURS_MATCH_I1, 0.0)
            
            # 更激进地调整相似度公式
            # 对于相似轮廓(ret接近0)会给出较高分数，但对于不同轮廓会迅速降低分数
            if ret < 0.01:  # 非常相似的轮廓
                similarity = 90 + (1 - ret * 100) * 10  # 90-100范围
            elif ret < 0.05:  # 较相似的轮廓
                similarity = 70 + (0.05 - ret) * 400  # 70-90范围
            elif ret < 0.1:  # 中等相似度
                similarity = 50 + (0.1 - ret) * 400  # 50-70范围
            elif ret < 0.2:  # 较低相似度
                similarity = 30 + (0.2 - ret) * 200  # 30-50范围
            else:  # 非常不同的轮廓
                similarity = max(0, 30 - (ret - 0.2) * 100)  # 0-30范围
            
            # 确保结果在0-100范围内
            similarity = min(100, max(0, similarity))
            
            return similarity
            
        except Exception as e:
            self.logger.error(f"Error comparing contours: {e}")
            return 0
    
    def visualize_match(self, frame, hand_contour, template_contour, match_result):
        """
        可视化显示匹配结果
        
        参数:
            frame (numpy.ndarray): 输入帧
            hand_contour (numpy.ndarray): 手部轮廓
            template_contour (numpy.ndarray): 模板轮廓
            match_result (dict): 匹配结果
            
        返回:
            numpy.ndarray: 带有可视化结果的帧
        """
        result = frame.copy()
        
        # 绘制手部轮廓
        cv.drawContours(result, [hand_contour], -1, (0, 255, 0), 2)
        
        # 准备模板轮廓用于显示
        # 需要调整模板轮廓位置，放在屏幕一角
        h, w = frame.shape[:2]
        template_display_size = min(w, h) // 3
        
        # 计算模板轮廓的边界框
        x, y, tw, th = cv.boundingRect(template_contour)
        
        # 计算缩放比例
        scale = min(template_display_size / tw, template_display_size / th) * 0.8
        
        # 创建变换矩阵
        M = np.array([
            [scale, 0, w - template_display_size],
            [0, scale, template_display_size // 4]
        ], dtype=np.float32)
        
        # 应用变换
        transformed_template = cv.transform(template_contour, M)
        
        # 绘制变换后的模板轮廓
        cv.drawContours(result, [transformed_template], -1, (255, 0, 0), 2)
        
        # 绘制匹配信息
        if match_result:
            # 匹配状态文本
            match_status = "Matched" if match_result["matched"] else "Not Matched"
            match_color = (0, 255, 0) if match_result["matched"] else (0, 0, 255)
            
            # 绘制匹配信息
            cv.putText(result, 
                      f"Template: {match_result['name']}", 
                      (10, 30), 
                      cv.FONT_HERSHEY_SIMPLEX, 
                      0.7, 
                      (255, 255, 255), 
                      2)
            
            cv.putText(result, 
                      f"Similarity: {match_result['similarity']:.1f}", 
                      (10, 60), 
                      cv.FONT_HERSHEY_SIMPLEX, 
                      0.7, 
                      (255, 255, 255), 
                      2)
            
            cv.putText(result, 
                      f"Status: {match_status}", 
                      (10, 90), 
                      cv.FONT_HERSHEY_SIMPLEX, 
                      0.7, 
                      match_color, 
                      2)
        
        return result