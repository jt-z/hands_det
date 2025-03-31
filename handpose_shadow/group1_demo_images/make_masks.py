import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def create_template_dir():
    """创建模板目录结构"""
    base_dir = "handpose_shadow/templates"
    group1_dir = os.path.join(base_dir, "group1")
    
    # 创建目录
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(group1_dir, exist_ok=True)
    
    return group1_dir

def process_hand_image(image_path, output_path, show_steps=False):
    """
    处理手部图像，提取轮廓并生成黑白模板
    
    参数:
        image_path: 输入图像路径
        output_path: 输出模板路径
        show_steps: 是否显示处理步骤
    """
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return False
    
    # 调整图像大小为512x512
    img = cv2.resize(img, (512, 512))
    
    # 转换到HSV颜色空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 设置肤色范围（可能需要根据图像调整）
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    # 创建肤色掩码
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # 应用形态学操作来改善掩码
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 创建空白画布（白底）
    canvas = np.ones((512, 512), dtype=np.uint8) * 255
    
    if contours:
        # 找到最大的轮廓（假设是手部）
        max_contour = max(contours, key=cv2.contourArea)
        
        # 填充轮廓（黑色填充在白色背景上）
        cv2.drawContours(canvas, [max_contour], -1, 0, -1)
    
    # 保存模板
    cv2.imwrite(output_path, canvas)
    
    if show_steps:
        # 显示处理步骤
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.title("原始图像")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        plt.subplot(2, 3, 2)
        plt.title("HSV 转换")
        plt.imshow(hsv)
        
        plt.subplot(2, 3, 3)
        plt.title("肤色掩码")
        plt.imshow(mask, cmap='gray')
        
        plt.subplot(2, 3, 4)
        contour_img = img.copy()
        cv2.drawContours(contour_img, [max_contour], -1, (0, 255, 0), 2)
        plt.title("检测到的手部轮廓")
        plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
        
        plt.subplot(2, 3, 5)
        plt.title("最终模板")
        plt.imshow(canvas, cmap='gray')
        
        plt.tight_layout()
        plt.show()
    
    return True

def process_all_images(input_paths, output_dir, output_names):
    """
    处理所有手部图像并生成黑白模板
    
    参数:
        input_paths: 输入图像路径列表
        output_dir: 输出目录
        output_names: 输出文件名列表
    """
    results = []
    
    for i, (input_path, output_name) in enumerate(zip(input_paths, output_names)):
        output_path = os.path.join(output_dir, output_name)
        print(f"处理图像 {i+1}/{len(input_paths)}: {input_path} -> {output_path}")
        
        success = process_hand_image(input_path, output_path, show_steps=(i==0))  # 仅显示第一张图片的处理步骤
        results.append((output_name, success))
    
    return results

def update_config_file(template_names):
    """
    生成更新config.py文件中TEMPLATE_GROUPS部分的代码
    
    参数:
        template_names: 模板名称列表，不包含扩展名
    """
    config_code = """
# 修改 config.py 中的 TEMPLATE_GROUPS 部分
TEMPLATE_GROUPS = {
    "group1": [  # 城市场景
        {"id": "1001", "file": os.path.join("group1", "human.png"), "name": "人", "threshold": 55},
        {"id": "1002", "file": os.path.join("group1", "dog.png"), "name": "狗", "threshold": 58},
        {"id": "1003", "file": os.path.join("group1", "weasel.png"), "name": "黄鼬", "threshold": 60},
        {"id": "1004", "file": os.path.join("group1", "hedgehog.png"), "name": "刺猬", "threshold": 57},
        {"id": "1005", "file": os.path.join("group1", "blackbird.png"), "name": "乌鸫", "threshold": 56},
    ],
    # 其他组保持不变...
}
    """
    
    print("配置文件更新示例代码:")
    print(config_code)

def main():
    # 输入图像路径（这里需要修改为你的图像路径）
    input_paths = [
        "human.jpg",
        "dog.jpg",
        "weasel.jpg",
        "hedgehog.jpg",
        "blackbird.jpg"
    ]
    
    # 输出文件名
    output_names = [
        "human.png",   # 人
        "dog.png",     # 狗
        "weasel.png",  # 黄鼬
        "hedgehog.png", # 刺猬
        "blackbird.png" # 乌鸫
    ]
    
    # 创建模板目录
    output_dir = create_template_dir()
    
    # 处理所有图像
    results = process_all_images(input_paths, output_dir, output_names)
    
    # 打印处理结果
    print("\n处理结果:")
    for name, success in results:
        status = "成功" if success else "失败"
        print(f"- {name}: {status}")
    
    # 生成配置文件更新代码
    template_names = [os.path.splitext(name)[0] for name in output_names]
    update_config_file(template_names)
    
    print("\n所有模板已生成完毕！")

if __name__ == "__main__":
    main()