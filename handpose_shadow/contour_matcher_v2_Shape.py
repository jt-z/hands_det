"""
轮廓匹配模块 - 基于形状上下文算法
负责比较手部轮廓与模板轮廓，计算相似度
"""

import cv2 as cv
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from .config import SIMILARITY_THRESHOLD
from .utils.logging_utils import get_logger, LogPerformance

class ContourMatcher:
    """轮廓匹配器类，使用形状上下文算法比较手部轮廓与模板轮廓"""
    
    def __init__(self, default_threshold=None):
        """
        初始化轮廓匹配器
        
        参数:
            default_threshold (float, 可选): 默认相似度阈值，默认使用配置值
        """
        self.logger = get_logger("contour_matcher")
        
        # 使用传入的阈值或默认配置
        self.default_threshold = default_threshold or SIMILARITY_THRESHOLD
        
        # 形状上下文参数
        self.n_points = 64  # 轮廓采样点数，针对手部优化
        self.r_bins = 5     # 距离bins数量
        self.theta_bins = 12 # 角度bins数量
        
        # 距离和角度bins设置
        self.r_bins_edges = np.logspace(-1.2, 0.6, self.r_bins + 1)  # 针对手部距离范围优化
        self.theta_bins_edges = np.linspace(-np.pi, np.pi, self.theta_bins + 1)
        
        self.logger.info(f"ContourMatcher initialized with Shape Context algorithm")
        self.logger.info(f"Parameters: n_points={self.n_points}, r_bins={self.r_bins}, theta_bins={self.theta_bins}")
    
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
            if similarity > 0:
                print(f"Shape Context similarity: {similarity:.2f}")
            
            self_Debug = False
            if self_Debug:
                from handpose_shadow.utils.debug_utils import visualize_contour
                visualize_contour(template_contour)
                visualize_contour(hand_contour)

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
        使用形状上下文算法比较两个轮廓的相似度
        
        参数:
            contour1 (numpy.ndarray): 模板轮廓
            contour2 (numpy.ndarray): 手部轮廓
            
        返回:
            float: 相似度分数 (0-100)
        """
        try:
            # 输入验证
            if contour1 is None or contour2 is None:
                self.logger.warning("Cannot compare: one of the contours is None")
                return 0
                
            if len(contour1) < 3 or len(contour2) < 3:
                self.logger.warning("Cannot compare: contour has less than 3 points")
                return 0
                
            # 检查轮廓面积
            area1 = cv.contourArea(contour1)
            area2 = cv.contourArea(contour2)
            if area1 <= 0 or area2 <= 0:
                self.logger.warning(f"Invalid contour area: {area1}, {area2}")
                return 0
            
            # 快速几何特征筛选
            if not self._quick_geometric_filter(contour1, contour2):
                return 0
            
            # 使用形状上下文算法计算相似度
            similarity = self._shape_context_matching(contour1, contour2)
            
            # 确保结果在0-100范围内
            similarity = min(100, max(0, similarity))
            
            return similarity
            
        except Exception as e:
            self.logger.error(f"Error comparing contours with Shape Context: {e}")
            return 0
    
    def _quick_geometric_filter(self, contour1, contour2, area_ratio_range=(0.3, 3.0), perimeter_ratio_range=(0.4, 2.5)):
        """
        快速几何特征筛选，排除明显不匹配的轮廓
        
        参数:
            contour1, contour2: 输入轮廓
            area_ratio_range: 面积比例允许范围
            perimeter_ratio_range: 周长比例允许范围
            
        返回:
            bool: 是否通过筛选
        """
        try:
            # 面积比较
            area1 = cv.contourArea(contour1)
            area2 = cv.contourArea(contour2)
            area_ratio = area2 / area1 if area1 > 0 else 0
            
            if not (area_ratio_range[0] <= area_ratio <= area_ratio_range[1]):
                return False
            
            # 周长比较
            perimeter1 = cv.arcLength(contour1, True)
            perimeter2 = cv.arcLength(contour2, True)
            perimeter_ratio = perimeter2 / perimeter1 if perimeter1 > 0 else 0
            
            if not (perimeter_ratio_range[0] <= perimeter_ratio <= perimeter_ratio_range[1]):
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Error in geometric filter: {e}")
            return True  # 出错时不过滤
    
    def _resample_contour(self, contour, n_points):
        """
        重采样轮廓到固定点数
        
        参数:
            contour (numpy.ndarray): 输入轮廓
            n_points (int): 目标点数
            
        返回:
            numpy.ndarray: 重采样后的轮廓点
        """
        # 确保轮廓是正确的形状
        if len(contour.shape) == 3:
            contour = contour.reshape(-1, 2)
        
        # 计算轮廓周长
        perimeter = cv.arcLength(contour, True)
        if perimeter <= 0:
            return None
        
        # 计算每个采样点之间的弧长距离
        step = perimeter / n_points
        
        # 沿轮廓等距采样
        sampled_points = []
        current_length = 0
        target_length = 0
        
        for i in range(len(contour)):
            current_point = contour[i]
            next_point = contour[(i + 1) % len(contour)]
            
            # 计算当前段的长度
            segment_length = np.linalg.norm(next_point - current_point)
            
            # 检查是否需要在当前段内采样
            while target_length <= current_length + segment_length and len(sampled_points) < n_points:
                if segment_length > 0:
                    # 计算采样点在当前段的位置
                    t = (target_length - current_length) / segment_length
                    sampled_point = current_point + t * (next_point - current_point)
                    sampled_points.append(sampled_point)
                else:
                    sampled_points.append(current_point)
                
                target_length += step
            
            current_length += segment_length
        
        # 确保采样到足够的点
        while len(sampled_points) < n_points:
            sampled_points.append(contour[-1])
        
        return np.array(sampled_points[:n_points])
    
    def _normalize_contour(self, contour):
        """
        轮廓归一化：中心化和尺度归一化
        
        参数:
            contour (numpy.ndarray): 输入轮廓
            
        返回:
            numpy.ndarray: 归一化后的轮廓
        """
        # 计算质心
        moments = cv.moments(contour)
        if moments['m00'] == 0:
            return contour
        
        cx = moments['m10'] / moments['m00']
        cy = moments['m01'] / moments['m00']
        
        # 中心化
        centered = contour - np.array([cx, cy])
        
        # 尺度归一化
        distances = np.sqrt(np.sum(centered**2, axis=1))
        max_distance = np.max(distances)
        
        if max_distance > 0:
            normalized = centered / max_distance
        else:
            normalized = centered
        
        return normalized
    
    def _compute_shape_context(self, points):
        """
        计算形状上下文描述子
        
        参数:
            points (numpy.ndarray): 轮廓点集
            
        返回:
            numpy.ndarray: 形状上下文描述子矩阵 (n_points, n_bins)
        """
        n = len(points)
        if n == 0:
            return np.array([])
        
        # 计算点之间的距离矩阵
        dist_matrix = cdist(points, points)
        
        # 计算角度矩阵
        angle_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = points[j, 0] - points[i, 0]
                    dy = points[j, 1] - points[i, 1]
                    angle_matrix[i, j] = np.arctan2(dy, dx)
        
        # 为每个点计算形状上下文
        shape_contexts = []
        
        for i in range(n):
            # 排除自身点
            valid_indices = np.arange(n) != i
            distances = dist_matrix[i, valid_indices]
            angles = angle_matrix[i, valid_indices]
            
            # 归一化距离
            mean_dist = np.mean(distances)
            if mean_dist > 0:
                distances = distances / mean_dist
            
            # 构建2D直方图
            hist, _, _ = np.histogram2d(
                distances, 
                angles, 
                bins=[self.r_bins_edges, self.theta_bins_edges]
            )
            
            # 展平并归一化直方图
            hist = hist.flatten()
            hist_sum = np.sum(hist)
            if hist_sum > 0:
                hist = hist / hist_sum
            
            shape_contexts.append(hist)
        
        return np.array(shape_contexts)
    
    def _shape_context_matching(self, contour1, contour2):
        """
        使用形状上下文进行轮廓匹配
        
        参数:
            contour1, contour2: 输入轮廓
            
        返回:
            float: 相似度分数 (0-100)
        """
        try:
            # 归一化轮廓
            norm_contour1 = self._normalize_contour(contour1.reshape(-1, 2))
            norm_contour2 = self._normalize_contour(contour2.reshape(-1, 2))
            
            # 重采样到固定点数
            points1 = self._resample_contour(norm_contour1, self.n_points)
            points2 = self._resample_contour(norm_contour2, self.n_points)
            
            if points1 is None or points2 is None or len(points1) == 0 or len(points2) == 0:
                return 0
            
            # 计算形状上下文
            sc1 = self._compute_shape_context(points1)
            sc2 = self._compute_shape_context(points2)
            
            if sc1.size == 0 or sc2.size == 0:
                return 0
            
            # 使用卡方距离计算代价矩阵
            cost_matrix = self._chi2_distance_matrix(sc1, sc2)
            
            # 使用匈牙利算法找最佳匹配
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            # 计算总匹配成本
            total_cost = cost_matrix[row_indices, col_indices].sum()
            average_cost = total_cost / self.n_points
            
            # 转换为相似度分数 (0-100)
            # 经验公式：成本越低，相似度越高
            if average_cost < 0.1:
                similarity = 90 + (0.1 - average_cost) * 100  # 90-100范围
            elif average_cost < 0.3:
                similarity = 70 + (0.3 - average_cost) * 100  # 70-90范围
            elif average_cost < 0.5:
                similarity = 50 + (0.5 - average_cost) * 100  # 50-70范围
            elif average_cost < 1.0:
                similarity = 25 + (1.0 - average_cost) * 50   # 25-50范围
            else:
                similarity = max(0, 25 - (average_cost - 1.0) * 25)  # 0-25范围
            
            return similarity
            
        except Exception as e:
            self.logger.error(f"Error in shape context matching: {e}")
            return 0
    
    def _chi2_distance_matrix(self, sc1, sc2):
        """
        计算形状上下文之间的卡方距离矩阵
        
        参数:
            sc1, sc2: 形状上下文描述子矩阵
            
        返回:
            numpy.ndarray: 距离矩阵
        """
        def chi2_distance(hist1, hist2):
            """计算两个直方图之间的卡方距离"""
            # 避免除零
            eps = 1e-10
            sum_hist = hist1 + hist2 + eps
            diff_hist = hist1 - hist2
            
            # 卡方距离公式
            chi2_dist = 0.5 * np.sum((diff_hist ** 2) / sum_hist)
            return chi2_dist
        
        n1, n2 = len(sc1), len(sc2)
        cost_matrix = np.zeros((n1, n2))
        
        for i in range(n1):
            for j in range(n2):
                cost_matrix[i, j] = chi2_distance(sc1[i], sc2[j])
        
        return cost_matrix
    
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
                      f"Shape Context Similarity: {match_result['similarity']:.1f}", 
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