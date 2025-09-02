"""
基于Inception V3特征向量的图像匹配模块
使用预训练的Inception V3模型提取深度特征，通过余弦相似度进行图像匹配
"""

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2 as cv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from typing import Optional, Union
from .config import SIMILARITY_THRESHOLD  # 假设配置文件存在
from .utils.logging_utils import get_logger, LogPerformance  # 假设日志工具存在


class InceptionImageMatcher:
    """基于Inception V3特征向量的图像匹配器类"""
    
    def __init__(self, model_path: str = 'HSPR_InceptionV3.pt', 
                 default_threshold: Optional[float] = None,
                 device: Optional[str] = None):
        """
        初始化图像匹配器
        
        参数:
            model_path (str): 预训练模型权重文件路径
            default_threshold (float, 可选): 默认相似度阈值，默认使用配置值
            device (str, 可选): 计算设备 ('cuda', 'cpu' 或 None 自动选择)
        """
        self.logger = get_logger("inception_image_matcher")
        
        # 使用传入的阈值或默认配置
        self.default_threshold = default_threshold or SIMILARITY_THRESHOLD
        
        # 设备选择
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # 初始化模型
        self.model = None
        self.preprocess = None
        self._init_model(model_path)
        
        self.logger.info(f"InceptionImageMatcher initialized with device={self.device}, "
                        f"default_threshold={self.default_threshold}")
    
    def _init_model(self, model_path: str):
        """初始化Inception V3模型"""
        try:
            # 检查模型文件是否存在
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model weight file not found: {model_path}")
                
            self.logger.info(f"Loading model from {model_path}")
            
            # 1. 创建一个标准的InceptionV3模型，使用默认参数
            # 这里我们不用担心默认是1000类，因为我们不加载它的分类层
            self.model = models.inception_v3(weights=None, aux_logits=True)
            
            # 2. 从你的权重文件中加载完整的state_dict
            state_dict = torch.load(model_path, map_location=self.device)
            
            # 3. 筛选出我们需要的权重
            # 创建一个新的有序字典，只包含模型主干（特征提取部分）的权重
            new_state_dict = {}
            for key, value in state_dict.items():
                # 排除分类器层的权重
                if 'fc' not in key and 'AuxLogits' not in key:
                    new_state_dict[key] = value
            
            # 4. 将筛选后的权重加载到模型中
            # strict=False 允许部分权重不匹配，即我们忽略了分类器层
            missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)
            
            # 记录权重加载情况
            if missing_keys:
                self.logger.info(f"Missing keys (expected for fc/AuxLogits): {missing_keys}")
            if unexpected_keys:
                self.logger.warning(f"Unexpected keys in state_dict: {unexpected_keys}")
            
            # 5. 移除分类器层，只保留特征提取部分
            self.model.fc = torch.nn.Identity()
            
            # 设置为评估模式并移动到指定设备
            self.model.eval()
            self.model = self.model.to(self.device)
            
            # 初始化图像预处理流水线
            self.preprocess = transforms.Compose([
                transforms.Resize(299),  # InceptionV3输入尺寸
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),  # ImageNet标准化参数
            ])
            
            self.logger.info("Model structure and feature extractor weights loaded successfully!")
            self.logger.info(f"Loaded {len(new_state_dict)} weight parameters")
            
        except Exception as e:
            self.logger.error(f"Error initializing model: {e}")
            raise
    
    def _image_to_pil(self, image: Union[str, np.ndarray, Image.Image]) -> Image.Image:
        """
        将不同格式的图像转换为PIL Image对象
        
        参数:
            image: 图像路径字符串、numpy数组或PIL Image对象
            
        返回:
            PIL.Image: 转换后的PIL图像对象
        """
        if isinstance(image, str):
            # 文件路径
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image file not found: {image}")
            return Image.open(image).convert('RGB')
        
        elif isinstance(image, np.ndarray):
            # numpy数组 (OpenCV BGR格式)
            if len(image.shape) == 3 and image.shape[2] == 3:
                # BGR to RGB
                image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                return Image.fromarray(image_rgb)
            elif len(image.shape) == 3 and image.shape[2] == 1:
                # 灰度图
                return Image.fromarray(image.squeeze(), mode='L').convert('RGB')
            else:
                return Image.fromarray(image).convert('RGB')
        
        elif isinstance(image, Image.Image):
            # 已经是PIL图像
            return image.convert('RGB')
        
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
    
    def extract_features(self, image: Union[str, np.ndarray, Image.Image]) -> Optional[np.ndarray]:
        """
        从图像中提取Inception V3特征向量
        
        参数:
            image: 输入图像 (路径、numpy数组或PIL Image)
            
        返回:
            numpy.ndarray: 特征向量 或 None
        """
        try:
            # 转换为PIL图像
            pil_image = self._image_to_pil(image)
            
            # 预处理
            img_tensor = self.preprocess(pil_image).unsqueeze(0)  # 添加batch维度
            img_tensor = img_tensor.to(self.device)
            
            # 特征提取
            with torch.no_grad():
                features = self.model(img_tensor)
            
            # 处理可能的辅助输出
            if isinstance(features, tuple):
                features = features[0]
            
            # 转换为numpy数组并扁平化
            features_np = features.cpu().numpy().flatten()
            
            return features_np
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            return None
    
    @LogPerformance()
    def _compare_images(self, template_image: Union[str, np.ndarray, Image.Image], 
                       current_frame: Union[str, np.ndarray, Image.Image]) -> float:
        """
        比较模板图像和当前帧的相似度
        
        参数:
            template_image: 模板图像
            current_frame: 当前视频帧
            
        返回:
            float: 相似度分数 (0-100)
        """
        try:
            # 提取特征向量
            template_features = self.extract_features(template_image)
            frame_features = self.extract_features(current_frame)
            
            if template_features is None:
                self.logger.warning("Cannot extract features from template image")
                return 0.0
                
            if frame_features is None:
                self.logger.warning("Cannot extract features from current frame")
                return 0.0
            
            # 计算余弦相似度
            template_features_2d = template_features.reshape(1, -1)
            frame_features_2d = frame_features.reshape(1, -1)
            
            cosine_sim = cosine_similarity(template_features_2d, frame_features_2d)[0, 0]
            
            # 转换为0-100分数
            # 余弦相似度范围是[-1, 1]，映射到[0, 100]
            similarity_score = (cosine_sim + 1) * 50
            
            return float(similarity_score)
            
        except Exception as e:
            self.logger.error(f"Error comparing images: {e}")
            return 0.0
    
    def clear_cache(self):
        """清除缓存（保持接口一致性）"""
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.info("Cache cleared")
    
    def __del__(self):
        """析构函数"""
        # 清理GPU内存
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()