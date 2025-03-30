"""
手部检测器测试模块
"""

import unittest
import os
import sys
import cv2 as cv
import numpy as np

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from handpose_shadow.hand_detector import HandDetector

class TestHandDetector(unittest.TestCase):
    """测试手部检测器类"""
    
    # 修改test_hand_detector.py中的setUp方法
    def setUp(self):
        """测试准备工作"""
        # 创建测试图像
        self.skin_color = (5, 150, 150)  # 改为元组，不是np.array
        self.test_image = self._create_test_image()
        
        # 创建手部检测器
        self.hand_detector = HandDetector(
            skin_lower=[0, 130, 130],
            skin_upper=[10, 170, 170],
            min_area=500,
            resize_frame=False
        )
    
    def _create_test_image(self, width=640, height=480):
        """创建测试图像，包含模拟的手部区域"""
        # 创建黑色背景图像
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 在图像中央创建模拟的手部形状 (简单的矩形)
        x1, y1 = width // 4, height // 4
        x2, y2 = width * 3 // 4, height * 3 // 4
        
        # 将BGR转换到HSV，以便放置皮肤颜色
        img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        
        # 在HSV空间画一个皮肤色的矩形
        cv.rectangle(img_hsv, (x1, y1), (x2, y2), self.skin_color, -1)
        
        # 转回BGR
        img = cv.cvtColor(img_hsv, cv.COLOR_HSV2BGR)
        
        return img
    
    def test_detect_skin(self):
        """测试皮肤检测功能"""
        # 使用私有方法检测皮肤
        skin_mask = self.hand_detector._detect_skin(self.test_image)
        
        # 验证结果
        self.assertIsNotNone(skin_mask, "皮肤掩码不应为None")
        self.assertEqual(skin_mask.dtype, np.uint8, "掩码应为8位无符号整数")
        self.assertEqual(skin_mask.shape, (self.test_image.shape[0], self.test_image.shape[1]), "掩码尺寸应与图像相同")
        
        # 检查掩码是否检测到了模拟的手部区域
        x1, y1 = self.test_image.shape[1] // 4, self.test_image.shape[0] // 4
        x2, y2 = self.test_image.shape[1] * 3 // 4, self.test_image.shape[0] * 3 // 4
        
        # 计算矩形区域内白色像素的比例
        rect = skin_mask[y1:y2, x1:x2]
        white_percent = np.sum(rect == 255) / rect.size
        
        # 超过50%区域检测为皮肤
        self.assertGreater(white_percent, 0.5, "应检测到大部分模拟手部区域")
    
    def test_extract_contours(self):
        """测试轮廓提取功能"""
        # 创建测试掩码
        mask = np.zeros((480, 640), dtype=np.uint8)
        cv.rectangle(mask, (160, 120), (480, 360), 255, -1)
        
        # 提取轮廓
        contours = self.hand_detector._extract_contours(mask)
        
        # 验证结果
        self.assertIsNotNone(contours, "轮廓不应为None")
        self.assertEqual(len(contours), 1, "应该只有一个轮廓")
        
        # 计算面积并验证
        area = cv.contourArea(contours[0])
        expected_area = (480-160) * (360-120)
        self.assertEqual(area, expected_area, "轮廓面积应该匹配")
    
    def test_select_main_contour(self):
        """测试主轮廓选择功能"""
        # 创建两个不同面积的轮廓
        c1 = np.array([[[100, 100]], [[200, 100]], [[200, 200]], [[100, 200]]], dtype=np.int32)  # 10000平方像素
        c2 = np.array([[[300, 300]], [[400, 300]], [[400, 350]], [[300, 350]]], dtype=np.int32)  # 5000平方像素
        contours = [c1, c2]
        
        # 选择主轮廓
        main_contour = self.hand_detector._select_main_contour(contours)
        
        # 验证结果
        self.assertIsNotNone(main_contour, "主轮廓不应为None")
        
        # 应该选择面积最大的轮廓
        area1 = cv.contourArea(c1)
        area2 = cv.contourArea(c2)
        self.assertGreater(area1, area2, "c1面积应大于c2")
        self.assertEqual(cv.contourArea(main_contour), area1, "应选择面积最大的轮廓")
        
        # 测试面积过小的轮廓被过滤
        small_contour = np.array([[[10, 10]], [[20, 10]], [[20, 20]], [[10, 20]]], dtype=np.int32)  # 100平方像素
        contours = [small_contour]
        main_contour = self.hand_detector._select_main_contour(contours)
        self.assertIsNone(main_contour, "面积小于阈值的轮廓应被过滤")
        
        # 测试空轮廓列表
        main_contour = self.hand_detector._select_main_contour([])
        self.assertIsNone(main_contour, "空轮廓列表应返回None")
    
    def test_detect_hand(self):
        """测试手部检测功能"""
        # 检测手部
        mask, contour = self.hand_detector.detect_hand(self.test_image)
        
        # 验证结果
        self.assertIsNotNone(mask, "掩码不应为None")
        self.assertIsNotNone(contour, "轮廓不应为None")
        
        # 计算轮廓面积
        area = cv.contourArea(contour)
        
        # 估计的期望面积 (矩形区域)
        width, height = self.test_image.shape[1], self.test_image.shape[0]
        expected_area = (width // 2) * (height // 2)  # 中央矩形区域
        
        # 由于边缘模糊和处理，允许误差范围
        self.assertGreater(area, expected_area * 0.7, "轮廓面积应接近期望值")
        self.assertLess(area, expected_area * 1.3, "轮廓面积应接近期望值")
    
    def test_draw_detection(self):
        """测试检测结果绘制功能"""
        # 准备测试数据
        mask = np.zeros((480, 640), dtype=np.uint8)
        cv.rectangle(mask, (160, 120), (480, 360), 255, -1)
        
        contour = np.array([[[160, 120]], [[480, 120]], [[480, 360]], [[160, 360]]], dtype=np.int32)
        
        # 绘制检测结果
        result = self.hand_detector.draw_detection(self.test_image, mask, contour)
        
        # 验证结果
        self.assertIsNotNone(result, "结果图像不应为None")
        self.assertEqual(result.shape, self.test_image.shape, "结果图像尺寸应与输入相同")
        
        # 绘制结果应该包含非黑色像素 (轮廓线等)
        self.assertTrue(np.any(result != 0), "结果图像应包含非黑色像素")

if __name__ == '__main__':
    unittest.main()