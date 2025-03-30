"""
轮廓匹配器测试模块
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

from handpose_shadow.contour_matcher import ContourMatcher

class TestContourMatcher(unittest.TestCase):
    """测试轮廓匹配器类"""
    
    def setUp(self):
        """测试准备工作"""
        # 创建轮廓匹配器
        self.contour_matcher = ContourMatcher(default_threshold=60)
        
        # 创建测试轮廓
        self.contour1 = self._create_rectangle_contour(100, 100, 200, 200)  # 矩形
        self.contour2 = self._create_rectangle_contour(110, 110, 210, 210)  # 略微移动的矩形
        self.contour3 = self._create_circle_contour(150, 150, 50)  # 圆形
        
        # 创建测试模板
        self.templates = {
            'rectangle': {
                'id': 'rectangle',
                'name': '矩形',
                'contour': self.contour1,
                'threshold': 70
            },
            'circle': {
                'id': 'circle',
                'name': '圆形',
                'contour': self.contour3,
                'threshold': 65
            }
        }
    
    def _create_rectangle_contour(self, x1, y1, x2, y2):
        """创建矩形轮廓"""
        return np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.int32)
    
    def _create_circle_contour(self, cx, cy, r, points=20):
        """创建圆形轮廓"""
        contour = []
        for i in range(points):
            angle = 2 * np.pi * i / points
            x = int(cx + r * np.cos(angle))
            y = int(cy + r * np.sin(angle))
            contour.append([[x, y]])
        return np.array(contour, dtype=np.int32)
    
    def test_compare_contours(self):
        """测试轮廓比较功能"""
        # 相同轮廓比较
        similarity1 = self.contour_matcher._compare_contours(self.contour1, self.contour1)
        self.assertAlmostEqual(similarity1, 100, delta=0.1, msg="相同轮廓相似度应接近100")
        
        # 相似轮廓比较
        similarity2 = self.contour_matcher._compare_contours(self.contour1, self.contour2)
        self.assertGreater(similarity2, 80, "相似轮廓相似度应较高")
        
        # 不同轮廓比较
        similarity3 = self.contour_matcher._compare_contours(self.contour1, self.contour3)
        self.assertLess(similarity3, 70, "不同轮廓相似度应较低")
    
    def test_match_with_templates(self):
        """测试与模板匹配功能"""
        # 与矩形模板匹配
        match1 = self.contour_matcher.match_with_templates(self.contour2, self.templates)
        
        # 验证结果
        self.assertIsNotNone(match1, "应返回匹配结果")
        self.assertEqual(match1['id'], 'rectangle', "应匹配到矩形模板")
        self.assertGreater(match1['similarity'], 80, "相似度应较高")
        self.assertTrue(match1['matched'], "应判定为匹配成功")
        
        # 与圆形模板匹配
        circle_contour = self._create_circle_contour(155, 155, 48)  # 略微偏移的圆
        match2 = self.contour_matcher.match_with_templates(circle_contour, self.templates)
        
        # 验证结果
        self.assertIsNotNone(match2, "应返回匹配结果")
        self.assertEqual(match2['id'], 'circle', "应匹配到圆形模板")
        self.assertGreater(match2['similarity'], match2['threshold'], "相似度应超过阈值")
        
        # 无匹配的情况
        # 创建与两个模板都不相似的三角形轮廓
        triangle = np.array([[[100, 200]], [[150, 100]], [[200, 200]]], dtype=np.int32)
        match3 = self.contour_matcher.match_with_templates(triangle, self.templates)
        
        # 验证结果
        self.assertIsNotNone(match3, "应返回最佳匹配结果")
        self.assertFalse(match3['matched'], "应判定为匹配失败")
    
    def test_edge_cases(self):
        """测试边缘情况"""
        # 空轮廓
        result1 = self.contour_matcher.match_with_templates(None, self.templates)
        self.assertIsNone(result1, "空轮廓应返回None")
        
        # 空模板
        result2 = self.contour_matcher.match_with_templates(self.contour1, {})
        self.assertIsNone(result2, "空模板应返回None")
        
        # 模板缺少轮廓
        templates_without_contour = {
            'bad_template': {
                'id': 'bad_template',
                'name': '坏模板'
                # 没有contour键
            }
        }
        result3 = self.contour_matcher.match_with_templates(self.contour1, templates_without_contour)
        self.assertIsNone(result3, "缺少轮廓的模板应被跳过")
    
    def test_visualize_match(self):
        """测试匹配可视化功能"""
        # 创建测试帧
        frame = np.zeros((300, 400, 3), dtype=np.uint8)
        
        # 创建匹配结果
        match_result = {
            'id': 'rectangle',
            'name': '矩形',
            'similarity': 85.0,
            'threshold': 70,
            'matched': True
        }
        
        # 生成可视化结果
        result = self.contour_matcher.visualize_match(frame, self.contour1, self.contour3, match_result)
        
        # 验证结果
        self.assertIsNotNone(result, "可视化结果不应为None")
        self.assertEqual(result.shape, frame.shape, "结果图像尺寸应与输入相同")
        
        # 由于绘制了文本和轮廓，图像不应全黑
        self.assertTrue(np.any(result != 0), "结果图像应包含非黑色像素")

if __name__ == '__main__':
    unittest.main()