import os
import sys
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from handpose_shadow.template_manager import TemplateManager

# 配置 Matplotlib 使用中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

def cv2_add_chinese_text(img, text, position, font_path="C:/Windows/Fonts/simhei.ttf", font_size=20, text_color=(0, 0, 0)):
    """使用 PIL 在 OpenCV 图像上添加中文文本"""
    # 判断图像是灰度还是彩色
    if len(img.shape) == 2:
        img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    
    # 将 OpenCV 图像转换为 PIL 图像
    pil_img = Image.fromarray(img)
    
    # 创建绘图对象
    draw = ImageDraw.Draw(pil_img)
    
    # 加载字体
    font = ImageFont.truetype(font_path, font_size)
    
    # 绘制文本
    draw.text(position, text, font=font, fill=text_color)
    
    # 将 PIL 图像转换回 OpenCV 格式
    return np.array(pil_img)

def visualize_template(template_data, title):
    """可视化显示模板"""
    # 创建一个白色画布
    img = np.ones((400, 400, 3), dtype=np.uint8) * 255
    
    # 绘制轮廓
    contour = template_data["contour"]
    cv.drawContours(img, [contour], -1, (0, 0, 0), 2)
    
    # 添加中文文本
    img = cv2_add_chinese_text(img, f"{title}: {template_data['name']}", (10, 30), font_size=24)
    
    # 显示图像
    plt.figure(figsize=(6, 6))
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
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
    print("\n生成自定义调试图像...")
    
    # 创建白色背景图像
    grid_size = 3  # 一行显示3个
    cell_size = 200
    rows = (len(templates) + grid_size - 1) // grid_size  # 向上取整
    
    canvas = np.ones((rows * cell_size, grid_size * cell_size, 3), dtype=np.uint8) * 255
    
    # 放置每个模板
    for i, (template_id, template) in enumerate(templates.items()):
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
        canvas = cv2_add_chinese_text(
            canvas, 
            template["name"], 
            (col * cell_size + 10, row * cell_size + 30),
            font_size=16
        )
    
    # 添加标题
    canvas = cv2_add_chinese_text(
        canvas, 
        "Group1 模板", 
        (10, 20),
        font_size=28
    )
    
    # 显示图像
    plt.figure(figsize=(10, 10))
    plt.imshow(cv.cvtColor(canvas, cv.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    
    # 可选：单独可视化每个模板
    for template_id, template in templates.items():
        visualize_template(template, template_id)

if __name__ == "__main__":
    main()