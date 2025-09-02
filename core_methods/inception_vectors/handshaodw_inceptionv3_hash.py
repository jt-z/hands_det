import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

# --- 步骤 1 & 2: 加载模型和权重 ---
# 确保你的 .pt 权重文件在当前目录下
model_path = 'HSPR_InceptionV3.pt'
if not os.path.exists(model_path):
    print(f"错误：未找到模型权重文件: {model_path}")
    print("请确保文件位于正确路径。")
else:
    print("成功找到模型权重文件，开始加载...")

    try:
        # 加载InceptionV3模型结构，注意pretrained=False，因为我们要加载自己的权重
        # aux_logits=False 确保我们不使用辅助分类器，简化操作
        model = models.inception_v3(pretrained=False, aux_logits=False)
        
        # 加载自定义权重
        # map_location='cpu' 确保即使在没有GPU的环境下也能加载
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        
        print("模型结构和权重加载成功！")
    except Exception as e:
        print(f"加载模型或权重时出错：{e}")

# --- 步骤 3: 移除分类器层 ---
# InceptionV3的最后分类层是名为 'fc' 的模块
# 我们可以直接将其替换为一个 'Identity' 模块，它会原样返回输入
model.fc = torch.nn.Identity()

# --- 步骤 4: 设置为评估模式 ---
# 这非常重要，它会禁用Dropout和Batch Normalization的训练行为
model.eval()
print("模型已设置为评估模式，分类器层已移除。")


# --- 步骤 5: 图像预处理流水线 ---
# InceptionV3的默认输入尺寸是 299x299
# 归一化参数是ImageNet的标准参数
preprocess = transforms.Compose([
    transforms.Resize(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- 步骤 6: 编写特征提取函数 ---
def extract_features(image_path_or_pil_image):
    """
    输入：图像文件路径或PIL图像对象
    输出：一个numpy数组，包含提取的特征向量
    """
    if isinstance(image_path_or_pil_image, str):
        img = Image.open(image_path_or_pil_image).convert('RGB')
    else:
        img = image_path_or_pil_image
        
    img_tensor = preprocess(img).unsqueeze(0)  # 添加一个batch维度
    
    with torch.no_grad(): # 禁用梯度计算，节省内存和时间
        features = model(img_tensor)
        
    # InceptionV3在forward中有一个辅助输出，但我们只关心主输出
    # 辅助输出是辅助分类器，在eval模式下通常不会使用，但为保险起见，我们只使用主输出
    if isinstance(features, tuple):
        features = features[0]
        
    # 将PyTorch张量转换为NumPy数组并扁平化
    return features.numpy().flatten()


# --- 示例：如何使用这个函数 ---
# 假设你已经有25个模板图像文件
template_paths = ["./templates/bird.jpg", "./templates/chicken.jpg", ...] # 替换为你的模板路径
template_features = []
for path in template_paths:
    feature = extract_features(path)
    template_features.append(feature)

# 将所有模板特征存储在一个NumPy数组中，方便快速计算
template_features_np = np.array(template_features)
print(f"成功提取并存储了 {template_features_np.shape[0]} 个模板的特征向量。")

# 现在你可以随时处理实时图像了
live_image_path = "./live_capture/current_frame.jpg" # 假设这是你的实时图片
live_features = extract_features(live_image_path)

# 计算余弦相似度
similarities = cosine_similarity([live_features], template_features_np)

# 找到相似度最高的模板索引和分数
best_match_index = np.argmax(similarities)
best_match_score = similarities[0, best_match_index]

print(f"\n实时手影的最佳匹配索引是: {best_match_index}")
print(f"最高相似度分数是: {best_match_score:.4f}")
