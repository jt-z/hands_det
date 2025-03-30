"""
模板管理模块
负责加载、处理和管理手影模板
"""

import os
import cv2 as cv
import numpy as np
from .config import TEMPLATES_DIR, TEMPLATE_GROUPS
from .utils.logging_utils import get_logger, LogPerformance

class TemplateManager:
    """模板管理器类，负责加载和管理手影模板"""
    
    def __init__(self, templates_dir=None, template_groups=None):
        """
        初始化模板管理器
        
        参数:
            templates_dir (str, 可选): 模板目录，默认使用配置值
            template_groups (dict, 可选): 模板组定义，默认使用配置值
        """
        self.logger = get_logger("template_manager")
        
        # 使用传入的值或默认配置
        self.templates_dir = templates_dir or TEMPLATES_DIR
        self.template_groups = template_groups or TEMPLATE_GROUPS
        
        # 存储已加载的模板
        self.templates = {}
        
        self.logger.info(f"TemplateManager initialized with templates_dir={self.templates_dir}")
        self.logger.info(f"Available template groups: {list(self.template_groups.keys())}")
    
    @LogPerformance()
    def load_group(self, group_name):
        """
        加载指定组的所有模板
        
        参数:
            group_name (str): 模板组名称
            
        返回:
            dict: 加载的模板字典，如果组不存在则返回空字典
        """
        if group_name not in self.template_groups:
            self.logger.error(f"Template group '{group_name}' not found")
            return {}
        
        self.logger.info(f"Loading template group: {group_name}")
        
        group_templates = {}
        templates_config = self.template_groups[group_name]
        
        for template_config in templates_config:
            template_id = template_config["id"]
            file_path = template_config["file"]
            name = template_config.get("name", template_id)
            threshold = template_config.get("threshold")
            
            try:
                # 构建完整的文件路径
                full_path = os.path.join(self.templates_dir, file_path)
                
                # 加载并处理模板图像
                template_contour = self._load_template(full_path)
                
                if template_contour is None:
                    self.logger.warning(f"Failed to load template: {file_path}")
                    continue
                
                # 存储模板信息
                group_templates[template_id] = {
                    "id": template_id,
                    "name": name,
                    "contour": template_contour,
                    "threshold": threshold,
                    "path": full_path
                }
                
                self.logger.debug(f"Loaded template: {name} ({template_id})")
                
            except Exception as e:
                self.logger.error(f"Error loading template {template_id}: {e}")
        
        # 存储到已加载模板字典中
        self.templates[group_name] = group_templates
        
        self.logger.info(f"Loaded {len(group_templates)} templates for group '{group_name}'")
        
        return group_templates
    
    def get_group(self, group_name):
        """
        获取指定组的模板，如果尚未加载则加载
        
        参数:
            group_name (str): 模板组名称
            
        返回:
            dict: 模板字典，如果组不存在则返回空字典
        """
        # 如果组已加载，直接返回
        if group_name in self.templates and self.templates[group_name]:
            self.logger.debug(f"Using cached templates for group '{group_name}'")
            return self.templates[group_name]
        
        # 否则加载模板组
        return self.load_group(group_name)
    
    def get_template(self, group_name, template_id):
        """
        获取特定模板
        
        参数:
            group_name (str): 模板组名称
            template_id (str): 模板ID
            
        返回:
            dict: 模板信息，如果不存在则返回None
        """
        group = self.get_group(group_name)
        return group.get(template_id)
    
    def get_all_groups(self):
        """
        获取所有可用的模板组名称
        
        返回:
            list: 模板组名称列表
        """
        return list(self.template_groups.keys())
    
    # 修改template_manager.py中的_load_template方法
    def _load_template(self, file_path):
        """
        加载并处理模板图像，提取轮廓
        
        参数:
            file_path (str): 模板图像文件路径
            
        返回:
            numpy.ndarray: 处理后的模板轮廓，如果加载失败则返回None
        """
        # 检查文件是否存在
        if not os.path.exists(file_path):
            self.logger.error(f"Template file not found: {file_path}")
            return None
        
        try:
            # 加载图像
            template_img = cv.imread(file_path, cv.IMREAD_UNCHANGED)
            
            if template_img is None:
                self.logger.error(f"Failed to read image: {file_path}")
                return None
            
            self.logger.debug(f"Template image shape: {template_img.shape}")
            
            # 检查图像维度并处理相应的情况
            if len(template_img.shape) == 3:  # 彩色图像
                if template_img.shape[2] == 4:  # 带Alpha通道
                    # 分离Alpha通道
                    alpha = template_img[:, :, 3]
                    template_img = template_img[:, :, :3]
                    
                    # 使用Alpha通道作为掩码
                    _, mask = cv.threshold(alpha, 127, 255, cv.THRESH_BINARY)
                    template_img = cv.bitwise_and(template_img, template_img, mask=mask)
                
                # 转为灰度
                gray = cv.cvtColor(template_img, cv.COLOR_BGR2GRAY)
            else:  # 已经是灰度图像
                gray = template_img
            
            # 二值化
            _, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
            
            # 提取轮廓
            contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            
            # 获取最大轮廓
            if not contours:
                self.logger.warning(f"No contours found in template: {file_path}")
                return None
                
            main_contour = max(contours, key=cv.contourArea)
            
            # 对轮廓进行光滑处理，减少噪点
            epsilon = 0.001 * cv.arcLength(main_contour, True)
            approx_contour = cv.approxPolyDP(main_contour, epsilon, True)
            
            return approx_contour
            
        except Exception as e:
            self.logger.error(f"Error processing template {file_path}: {e}")
            return None
        
    def generate_debug_image(self, group_name):
        """
        生成包含组内所有模板的调试图像
        
        参数:
            group_name (str): 模板组名称
            
        返回:
            numpy.ndarray: 调试图像，如果组不存在则返回None
        """
        group = self.get_group(group_name)
        
        if not group:
            return None
        
        # 创建拼接图像
        templates_count = len(group)
        
        # 确定网格大小
        grid_size = int(np.ceil(np.sqrt(templates_count)))
        cell_size = 200  # 每个模板单元格大小
        
        # 创建画布
        canvas = np.ones((grid_size * cell_size, grid_size * cell_size, 3), dtype=np.uint8) * 255
        
        # 放置每个模板
        for i, (template_id, template) in enumerate(group.items()):
            row = i // grid_size
            col = i % grid_size
            
            # 模板轮廓
            contour = template["contour"]
            
            # 计算放置位置
            x = col * cell_size + cell_size // 2
            y = row * cell_size + cell_size // 2
            
            # 调整轮廓大小以适应单元格
            x_min, y_min, w, h = cv.boundingRect(contour)
            
            scale = 0.8 * min(cell_size / w, cell_size / h) if w > 0 and h > 0 else 1.0
            
            M = np.array([
                [scale, 0, x - scale * (x_min + w/2)],
                [0, scale, y - scale * (y_min + h/2)]
            ], dtype=np.float32)
            
            transformed_contour = cv.transform(contour, M)
            
            # 绘制轮廓
            cv.drawContours(canvas, [transformed_contour], -1, (0, 0, 0), 2)
            
            # 添加模板名称
            cv.putText(
                canvas, 
                template["name"], 
                (col * cell_size + 10, row * cell_size + 20), 
                cv.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 0, 0), 
                1
            )
        
        return canvas