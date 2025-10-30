"""
独立的工具脚本，用于为所有模板图像预先计算InceptionV3特征向量。

功能:
- 遍历 `config.py` 中定义的所有模板组 (TEMPLATE_GROUPS)。
- 加载每个模板的原始图像。
- 使用 `InceptionImageMatcher` 提取特征向量。
- 将每个特征向量保存为对应的 .npy 文件，存储在指定的输出目录中。

如何运行:
1. 将此脚本放置在项目的根目录（与 `handpose_shadow` 文件夹同级）。
2. 打开终端，导航到项目根目录。
3. 运行命令: `python precompute_features.py`
"""

import os
import sys
import argparse
import numpy as np
from tqdm import tqdm

# --- 路径设置 ---
# 确保脚本可以找到 `handpose_shadow` 模块。
# 假设此脚本位于项目根目录。
try:
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
 
    from handpose_shadow.config import TEMPLATE_GROUPS, TEMPLATES_DIR
    from handpose_shadow.inception_image_matcher import InceptionImageMatcher
except ImportError as e:
    print(f"错误: 无法导入项目模块。请确保此脚本位于项目的根目录。")
    print(f"导入错误: {e}")
    sys.exit(1)


def generate_features(model_path: str, output_dir: str):
    """
    遍历所有模板图像，提取特征并保存为 .npy 文件。
    """
    print(f"正在初始化 InceptionImageMatcher 模型: {model_path}")
    
    try:
        matcher = InceptionImageMatcher(
            model_path=model_path 
            # feature_cache_dir=output_dir
        )
    except Exception as e:
        print(f"\n错误: 初始化 InceptionImageMatcher 失败。")
        print(f"请检查模型文件 '{model_path}' 是否存在，以及PyTorch是否已正确安装。")
        print(f"详细信息: {e}")
        sys.exit(1)

    print(f"特征向量将被保存到: {os.path.abspath(output_dir)}")
    print("-" * 60)
    
    total_templates = sum(len(group) for group in TEMPLATE_GROUPS.values())
    
    # 使用 tqdm 创建一个可视化的进度条
    with tqdm(total=total_templates, unit="image", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
        for group_id, templates in TEMPLATE_GROUPS.items():
            pbar.set_description(f"正在处理组 '{group_id}'")
            for template_info in templates:
                template_id = template_info.get("id")
                image_relative_path = template_info.get("file")

                if not template_id or not image_relative_path:
                    pbar.write(f"[警告] 跳过无效的模板条目 (组: '{group_id}'): {template_info}")
                    pbar.update(1)
                    continue

                image_full_path = os.path.join(TEMPLATES_DIR, image_relative_path)
                
                pbar.set_postfix_str(f"{image_relative_path}")

                if not os.path.exists(image_full_path):
                    pbar.write(f"[失败] 图像文件未找到，已跳过: {image_full_path}")
                    pbar.update(1)
                    continue

                # 提取特征
                features = matcher.extract_features(image_full_path)
                
                if features is not None:
                    # 使用 matcher 内的函数获取标准化的输出路径
                    output_path = f'{template_id}.npy'
                    
                    # 保存特征向量为 .npy 文件
                    np.save(output_path, features)
                else:
                    pbar.write(f"[失败] 无法从 {image_full_path} 提取特征")
                
                pbar.update(1)

    print("-" * 60)
    print("所有模板的特征已成功生成！")


def main():
    parser = argparse.ArgumentParser(
        description="为手影识别系统的所有模板图像预先计算 InceptionV3 特征向量。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model",
        type=str,
        default="handpose_shadow/HSPR_InceptionV3.pt",
        help="InceptionV3 模型权重文件 (.pt) 的路径。"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="handpose_shadow/features_cache",
        help="用于保存生成的 .npy 特征文件的目录。"
    )
    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    generate_features(args.model, args.output_dir)


if __name__ == "__main__":
    main()