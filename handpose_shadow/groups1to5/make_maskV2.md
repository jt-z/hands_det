

使用说明： 使用 make_maskv2代码, 可能要调整输入输出目录，其他都还好。

这是一个手影模板生成器的Python代码，主要用于从原始动物图片生成黑白轮廓模板。下面是使用方法：

## 基本使用步骤

### 1. 准备输入数据
```
原始图片目录结构应该是：
D:\Documents\Onedrive\Documents\A_Dashboard\PartTime\HandPoseShadow\handpose_shadow\groups1to5\original_pics\
├── group1/
│   ├── human.jpg
│   ├── dog.jpg
│   ├── weasel.jpg
│   └── ...
├── group2/
│   ├── arctic_wolf.jpg
│   ├── reindeer.jpg
│   └── ...
└── ...
```

### 2. 修改路径配置
修改 `main()` 函数中的路径：
```python
input_base_dir = "你的输入图片目录路径"
output_base_dir = "你的输出模板目录路径"
```

### 3. 直接运行
```bash
python your_script_name.py
```

## 核心处理流程

代码的处理逻辑是：
1. **读取图像** → 调整为512x512尺寸
2. **HSV颜色空间转换** → 提取轮廓区域
3. **二值化处理** → 生成掩码
4. **形态学操作** → 优化轮廓
5. **轮廓提取** → 找到最大轮廓
6. **生成模板** → 白底黑色填充的PNG模板

## 关键参数调整

如果生成的模板效果不好，可以调整这些参数：

```python
# HSV颜色范围（第102-103行）
lower_bound = np.array([0, 20, 60], dtype=np.uint8)
upper_bound = np.array([70, 255, 255], dtype=np.uint8)

# 形态学操作内核大小（第111行）
kernel = np.ones((5, 5), np.uint8)

# 轮廓面积阈值（第125行）
if cv2.contourArea(max_contour) > 1000:
```

## 调试和优化

### 查看处理步骤
```python
# 在 main() 中设置为 True
generator.process_all_groups(show_samples=True)

# 或者单独处理某个组并显示步骤
generator.process_group("group1", show_first_image=True)
```

### 单张图片测试
```python
generator = TemplateGenerator(input_dir, output_dir)
success = generator.process_single_image(
    "path/to/input.jpg", 
    "path/to/output.png", 
    show_steps=True
)
```

## 输出结果

- **模板文件**：生成白底黑色轮廓的PNG模板
- **配置代码**：自动生成config.py的配置代码
- **处理报告**：显示每组处理成功/失败的统计

代码会自动处理5个场景组（城市、冻原、稀树草原、针叶林、雨林），每组包含5种动物的轮廓模板生成。

需要注意的是，由于不同图片的背景和动物颜色差异，可能需要根据实际效果调整HSV颜色范围参数。