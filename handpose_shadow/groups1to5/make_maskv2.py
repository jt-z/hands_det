import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path

class TemplateGenerator:
    """手影模板生成器"""
    
    def __init__(self, input_base_dir, output_base_dir):
        """
        初始化生成器
        
        参数:
            input_base_dir: 输入图像的基础目录
            output_base_dir: 输出模板的基础目录
        """
        self.input_base_dir = Path(input_base_dir)
        self.output_base_dir = Path(output_base_dir)
        
        # 组配置
        self.group_configs = {
            "group1": {
                "scene": "城市场景",
                "animals": {
                    "human.jpg": {"id": "1001", "name": "人", "threshold": 55},
                    "dog.jpg": {"id": "1002", "name": "狗", "threshold": 58},
                    "weasel.jpg": {"id": "1003", "name": "黄鼬", "threshold": 60},
                    "hedgehog.jpg": {"id": "1004", "name": "刺猬", "threshold": 57},
                    "blackbird.jpg": {"id": "1005", "name": "乌鸫", "threshold": 56},
                }
            },
            "group2": {
                "scene": "冻原场景",
                "animals": {
                    "arctic_wolf.jpg": {"id": "2001", "name": "北极狼", "threshold": 55},
                    "reindeer.jpg": {"id": "2002", "name": "驯鹿", "threshold": 58},
                    "ptarmigan.jpg": {"id": "2003", "name": "岩雷鸟", "threshold": 60},
                    "musk_ox.jpg": {"id": "2004", "name": "麝牛", "threshold": 57},
                    "arctic_hare.jpg": {"id": "2005", "name": "北极兔", "threshold": 56},
                }
            },
            "group3": {
                "scene": "稀树草原场景",
                "animals": {
                    "lion.jpg": {"id": "3001", "name": "狮子", "threshold": 55},
                    "gemsbok.jpg": {"id": "3002", "name": "高角羚", "threshold": 58},
                    "elephant.jpg": {"id": "3003", "name": "非洲草原象", "threshold": 60},
                    "buffalo.jpg": {"id": "3004", "name": "非洲水牛", "threshold": 57},
                    "giraffe.jpg": {"id": "3005", "name": "长颈鹿", "threshold": 56},
                }
            },
            "group4": {
                "scene": "针叶林场景",
                "animals": {
                    "tiger.jpg": {"id": "4001", "name": "东北虎", "threshold": 55},
                    "brown_bear.jpg": {"id": "4002", "name": "棕熊", "threshold": 58},
                    "marten.jpg": {"id": "4003", "name": "紫貂", "threshold": 60},
                    "snake.jpg": {"id": "4004", "name": "棕黑锦蛇", "threshold": 57},
                    "moose.jpg": {"id": "4005", "name": "驼鹿", "threshold": 56},
                }
            },
            "group5": {
                "scene": "雨林场景",
                "animals": {
                    "clouded_leopard.jpg": {"id": "5001", "name": "云豹", "threshold": 55},
                    "pygmy_marmoset.jpg": {"id": "5002", "name": "蜂猴", "threshold": 58},
                    "snake.jpg": {"id": "5003", "name": "天堂金花蛇", "threshold": 60},
                    "bat.jpg": {"id": "5004", "name": "果蝠", "threshold": 57},
                    "locust.jpg": {"id": "5005", "name": "蚱蜢", "threshold": 56},
                }
            }
        }
    
    def create_output_dirs(self):
        """创建输出目录结构"""
        self.output_base_dir.mkdir(exist_ok=True)
        
        for group_name in self.group_configs.keys():
            group_dir = self.output_base_dir / group_name
            group_dir.mkdir(exist_ok=True)
            print(f"创建目录: {group_dir}")
    

    def center_crop_square(self, img):
        """居中裁剪为正方形，以短边为准"""
        h, w = img.shape[:2]
        
        # 以短边为准确定正方形边长
        size = min(h, w)
        
        # 计算裁剪起始坐标（居中）
        start_x = (w - size) // 2
        start_y = (h - size) // 2
        
        # 裁剪正方形区域
        cropped = img[start_y:start_y + size, start_x:start_x + size]
        
        return cropped

    def process_single_image(self, image_path, output_path, show_steps=False):
        """
        处理单张手部图像，提取轮廓并生成黑白模板
        
        参数:
            image_path: 输入图像路径
            output_path: 输出模板路径
            show_steps: 是否显示处理步骤
        
        返回:
            bool: 处理是否成功
        """
        try:
            # 读取图像
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"无法读取图像: {image_path}")
                return False
            
            # 先居中裁剪为正方形
            img = self.center_crop_square(img)

            # 调整图像大小为512x512
            img = cv2.resize(img, (512, 512))
            
            # 转换到HSV颜色空间
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # 设置肤色范围（可能需要根据图像调整）
            # 对于动物轮廓，我们使用更宽泛的范围
            # lower_bound = np.array([0, 20, 50], dtype=np.uint8)
            # upper_bound = np.array([180, 255, 255], dtype=np.uint8)

            lower_bound = np.array([0, 20, 60], dtype=np.uint8)  # 70
            upper_bound = np.array([70, 255, 255], dtype=np.uint8) # 20   需要不停地调整参数，获得合适的模板，因为背景是深色，不是白色。
            
            # 创建掩码
            mask = cv2.inRange(hsv, lower_bound, upper_bound)
            
            # 如果掩码为空，尝试使用灰度阈值
            if cv2.countNonZero(mask) < 1000:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 应用形态学操作来改善掩码
            # kernel = np.ones((3, 3), np.uint8) # 5,5 
            # mask = cv2.erode(mask, kernel, iterations=1)
            # mask = cv2.dilate(mask, kernel, iterations=3)
            mask = cv2.GaussianBlur(mask, (5, 5), 0) # 5,5 
            
            # 查找轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 创建空白画布（白底）
            canvas = np.ones((512, 512), dtype=np.uint8) * 255
            
            if contours:
                # 找到最大的轮廓
                max_contour = max(contours, key=cv2.contourArea)
                
                # 确保轮廓足够大
                if cv2.contourArea(max_contour) > 1000:
                    # 填充轮廓（黑色填充在白色背景上）
                    cv2.drawContours(canvas, [max_contour], -1, 0, -1)
                else:
                    print(f"警告: {image_path} 的轮廓太小，可能识别不准确")
                    # 仍然保存，但标记为可能有问题
                    cv2.drawContours(canvas, [max_contour], -1, 0, -1)
            else:
                print(f"警告: {image_path} 未找到有效轮廓")
                return False
            
            # 保存模板
            cv2.imwrite(str(output_path), canvas)
            
            if show_steps:
                self._show_processing_steps(img, hsv, mask, canvas, max_contour if contours else None)
            
            return True
            
        except Exception as e:
            print(f"处理图像 {image_path} 时出错: {e}")
            return False
    
    def _show_processing_steps(self, img, hsv, mask, canvas, contour):
        """显示处理步骤"""
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.title("原始图像")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.subplot(2, 3, 2)
        plt.title("HSV 转换")
        plt.imshow(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
        plt.axis('off')
        
        plt.subplot(2, 3, 3)
        plt.title("二值掩码")
        plt.imshow(mask, cmap='gray')
        plt.axis('off')
        
        if contour is not None:
            plt.subplot(2, 3, 4)
            contour_img = img.copy()
            cv2.drawContours(contour_img, [contour], -1, (0, 255, 0), 3)
            plt.title("检测到的轮廓")
            plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
        
        plt.subplot(2, 3, 5)
        plt.title("最终模板")
        plt.imshow(canvas, cmap='gray')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def process_group(self, group_name, show_first_image=False):
        """
        处理指定组的所有图像
        
        参数:
            group_name: 组名
            show_first_image: 是否显示第一张图片的处理步骤
        
        返回:
            dict: 处理结果统计
        """
        if group_name not in self.group_configs:
            print(f"未知的组名: {group_name}")
            return {}
        
        config = self.group_configs[group_name]
        input_group_dir = self.input_base_dir / group_name
        output_group_dir = self.output_base_dir / group_name
        
        print(f"\n=== 处理 {group_name} ({config['scene']}) ===")
        
        results = {}
        total_files = len(config['animals'])
        
        for i, (filename, animal_config) in enumerate(config['animals'].items()):
            input_path = input_group_dir / filename
            output_filename = filename.replace('.jpg', '.png')
            output_path = output_group_dir / output_filename
            
            print(f"[{i+1}/{total_files}] 处理: {filename} -> {output_filename}")
            
            if not input_path.exists():
                print(f"  错误: 文件不存在 {input_path}")
                results[filename] = False
                continue
            
            # 显示第一张图片的处理步骤
            show_steps = show_first_image and i == 0
            success = self.process_single_image(input_path, output_path, show_steps)
            
            results[filename] = success
            status = "成功" if success else "失败"
            print(f"  结果: {status}")
        
        return results
    
    def process_all_groups(self, show_samples=True):
        """
        处理所有组的图像
        
        参数:
            show_samples: 是否显示每组第一张图片的处理步骤
        """
        print("开始批量处理所有组的模板...")
        
        # 创建输出目录
        self.create_output_dirs()
        
        all_results = {}
        
        for group_name in self.group_configs.keys():
            results = self.process_group(group_name, show_first_image=show_samples)
            all_results[group_name] = results
        
        # 打印总结
        self.print_summary(all_results)
        
        # 生成配置代码
        self.generate_config_code()
    
    def print_summary(self, all_results):
        """打印处理结果总结"""
        print("\n" + "="*60)
        print("处理结果总结:")
        print("="*60)
        
        total_success = 0
        total_files = 0
        
        for group_name, results in all_results.items():
            config = self.group_configs[group_name]
            success_count = sum(1 for success in results.values() if success)
            total_count = len(results)
            
            total_success += success_count
            total_files += total_count
            
            print(f"\n{group_name} ({config['scene']}):")
            print(f"  成功: {success_count}/{total_count}")
            
            # 显示失败的文件
            failed_files = [filename for filename, success in results.items() if not success]
            if failed_files:
                print(f"  失败的文件: {', '.join(failed_files)}")
        
        print(f"\n总计: {total_success}/{total_files} 个文件处理成功")
        success_rate = (total_success / total_files * 100) if total_files > 0 else 0
        print(f"成功率: {success_rate:.1f}%")
    
    def generate_config_code(self):
        """生成config.py配置代码"""
        print("\n" + "="*60)
        print("config.py 配置代码:")
        print("="*60)
        
        config_lines = ['TEMPLATE_GROUPS = {']
        
        for group_name, config in self.group_configs.items():
            config_lines.append(f'    "{group_name}": [  # {config["scene"]}')
            
            for filename, animal_config in config['animals'].items():
                output_filename = filename.replace('.jpg', '.png')
                line = (f'        {{"id": "{animal_config["id"]}", '
                       f'"file": os.path.join("{group_name}", "{output_filename}"), '
                       f'"name": "{animal_config["name"]}", '
                       f'"threshold": {animal_config["threshold"]}}},')
                config_lines.append(line)
            
            config_lines.append('    ],')
        
        config_lines.append('}')
        
        print('\n'.join(config_lines))


def main():
    """主函数"""
    # 设置路径

    # 修改路径，改成更新后的代码路径。
    # input_base_dir = "D:\Documents\Onedrive\Documents\A_Dashboard\PartTime\HandPoseShadow\handpose_shadow\groups1to5\original_pics"
    # input_base_dir = "D:\Onedrive\Documents\A_Dashboard\PartTime\HandPoseShadow\handpose_shadow\groups1to5\original_pics"
    input_base_dir = "D:\dev_handshadow\\2025_2\\20250801_new_english"
    output_base_dir = ".\outputv2"
    
    # 创建生成器
    generator = TemplateGenerator(input_base_dir, output_base_dir)
    
    # 处理所有组
    generator.process_all_groups(show_samples=False)  # 设置为True可显示处理步骤图
    
    print("\n所有模板生成完毕！")
    print(f"输出目录: {output_base_dir}")
    print("\n请将生成的配置代码复制到 handpose_shadow/config.py 文件中的 TEMPLATE_GROUPS 部分。")


if __name__ == "__main__":
    main()