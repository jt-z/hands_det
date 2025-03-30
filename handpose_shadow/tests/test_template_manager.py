"""
模板管理器测试模块
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


import unittest
import os
import sys
import cv2 as cv
import numpy as np

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from handpose_shadow.template_manager import TemplateManager

class TestTemplateManager(unittest.TestCase):
    """测试模板管理器类"""
    
    def setUp(self):
        """测试准备工作"""
        # 创建测试目录
        self.test_dir = os.path.join(os.path.dirname(__file__), 'test_templates')
        os.makedirs(self.test_dir, exist_ok=True)
        
        # 创建测试组目录
        self.test_group_dir = os.path.join(self.test_dir, 'test_group')
        os.makedirs(self.test_group_dir, exist_ok=True)
        
        # 创建测试模板图像
        self.test_template_path = os.path.join(self.test_group_dir, 'test_template.png')
        self._create_test_template(self.test_template_path)
        
        # 配置测试模板组
        self.test_template_groups = {
            'test_group': [
                {'id': 'test_template', 'file': os.path.join('test_group', 'test_template.png'), 'name': '测试模板', 'threshold': 60}
            ]
        }
        
        # 创建模板管理器
        self.template_manager = TemplateManager(templates_dir=self.test_dir, template_groups=self.test_template_groups)
    
    def tearDown(self):
        """测试清理工作"""
        # 删除测试文件
        if os.path.exists(self.test_template_path):
            os.remove(self.test_template_path)
        
        # 删除测试目录
        if os.path.exists(self.test_group_dir):
            os.rmdir(self.test_group_dir)
        
        if os.path.exists(self.test_dir):
            os.rmdir(self.test_dir)
    
    def _create_test_template(self, path):
        """创建测试模板图像"""
        # 创建简单的矩形图像
        img = np.zeros((200, 200), dtype=np.uint8)
        cv.rectangle(img, (50, 50), (150, 150), 255, -1)
        cv.imwrite(path, img)
    
    def test_load_group(self):
        """测试加载模板组"""
        # 加载测试组
        templates = self.template_manager.load_group('test_group')
        
        # 验证结果
        self.assertEqual(len(templates), 1, "应该加载一个模板")
        self.assertIn('test_template', templates, "应该包含测试模板")
        self.assertEqual(templates['test_template']['name'], '测试模板', "模板名称不匹配")
        self.assertEqual(templates['test_template']['threshold'], 60, "模板阈值不匹配")
        self.assertIsNotNone(templates['test_template']['contour'], "应该包含轮廓")
    
    def test_get_group(self):
        """测试获取模板组"""
        # 获取测试组
        templates = self.template_manager.get_group('test_group')
        
        # 验证结果
        self.assertEqual(len(templates), 1, "应该获取一个模板")
        
        # 二次获取，应使用缓存
        templates_cached = self.template_manager.get_group('test_group')
        self.assertIs(templates, templates_cached, "二次获取应使用缓存")
    
    def test_get_nonexistent_group(self):
        """测试获取不存在的组"""
        templates = self.template_manager.get_group('nonexistent_group')
        self.assertEqual(len(templates), 0, "不存在的组应返回空字典")
    
    def test_get_template(self):
        """测试获取特定模板"""
        # 获取特定模板
        template = self.template_manager.get_template('test_group', 'test_template')
        
        # 验证结果
        self.assertIsNotNone(template, "应找到测试模板")
        self.assertEqual(template['name'], '测试模板', "模板名称不匹配")
        
        # 获取不存在的模板
        template = self.template_manager.get_template('test_group', 'nonexistent_template')
        self.assertIsNone(template, "不存在的模板应返回None")
    
    def test_get_all_groups(self):
        """测试获取所有组"""
        groups = self.template_manager.get_all_groups()
        self.assertEqual(len(groups), 1, "应返回一个组")
        self.assertIn('test_group', groups, "应包含测试组")
    
    def test_generate_debug_image(self):
        """测试生成调试图像"""
        # 加载组
        self.template_manager.load_group('test_group')
        
        # 生成调试图像
        debug_img = self.template_manager.generate_debug_image('test_group')
        
        # 验证结果
        self.assertIsNotNone(debug_img, "应生成调试图像")
        self.assertEqual(len(debug_img.shape), 3, "应为彩色图像")
        self.assertTrue(debug_img.shape[0] > 0 and debug_img.shape[1] > 0, "图像尺寸应大于0")

if __name__ == '__main__':
    unittest.main()