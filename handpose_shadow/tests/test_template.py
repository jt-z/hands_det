import os
import sys
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from handpose_shadow.template_manager import TemplateManager

def visualize_template(template_data, title):
    """可视化显示模板"""
    # 创建一个黑色画布 
    img = np.zeros((400, 400), dtype=np.uint8)
    
    # 绘制轮廓
    contour = template_data["contour"]
    cv.drawContours(img, [contour], -1, 255, 2)
    
    # 显示图像
    plt.figure(figsize=(6, 6))
    plt.title(f"{title}: {template_data['name']}")
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()

def main():
    # 创建模板管理器
    template_manager = TemplateManager()
    
    # 加载 group1
    print("加载 group1 模板...")
    templates = template_manager.load_group("group1")
    
    # 输出加载结果
    print(f"加载了 {len(templates)} 个模板")
    for template_id, template in templates.items():
        print(f"- {template_id}: {template['name']}")
        
        # 检查轮廓是否正确加载
        if template['contour'] is not None:
            print(f"  轮廓点数: {len(template['contour'])}")
            print(f"  轮廓面积: {cv.contourArea(template['contour'])}")
        else:
            print("  轮廓加载失败!")
    
    # 生成并显示调试图像
    print("\n生成调试图像...")
    debug_img = template_manager.generate_debug_image("group1")
    
    if debug_img is not None:
        plt.figure(figsize=(10, 10))
        plt.title("Group1 模板")
        plt.imshow(cv.cvtColor(debug_img, cv.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
    else:
        print("调试图像生成失败!")
    
    # 可选：单独可视化每个模板
    for template_id, template in templates.items():
        visualize_template(template, template_id)

if __name__ == "__main__":
    main()