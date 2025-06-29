import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
import time
import copy
import json

# 设置随机种子以确保可重复性
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

class HandShadowDataset(Dataset):
    def __init__(self, images_paths, labels, transform=None):
        self.image_paths = images_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像: {image_path}")
            # 创建一个空图像作为替代
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # 预处理图像（手部分割）
        processed_img = self.preprocess_hand_image(image)
        
        # 转换为PIL Image以便使用torchvision的transforms
        processed_img = Image.fromarray(processed_img)
        
        # 应用变换
        if self.transform:
            processed_img = self.transform(processed_img)
        
        return processed_img, label
    
    @staticmethod
    def preprocess_hand_image(image):
        """预处理手影图像，分割出手部并优化"""
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 二值化提取手部（使用自适应阈值可能效果更好）
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 形态学运算优化掩码
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # 找到轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # 获取最大轮廓
            max_contour = max(contours, key=cv2.contourArea)
            
            # 创建新的掩码
            mask = np.zeros_like(thresh)
            cv2.drawContours(mask, [max_contour], -1, 255, -1)
            
            # 应用掩码
            segmented = cv2.bitwise_and(gray, gray, mask=mask)
            
            # 标准化并转为三通道
            segmented = cv2.normalize(segmented, None, 0, 255, cv2.NORM_MINMAX)
            segmented_3ch = cv2.cvtColor(segmented, cv2.COLOR_GRAY2RGB)
            
            return segmented_3ch
        
        # 如果没有找到轮廓，返回原始灰度图（三通道）
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

# 定义数据增强变换
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 定义手影识别模型
class HandShadowNet(nn.Module):
    def __init__(self, num_classes):
        super(HandShadowNet, self).__init__()
        # 使用预训练的MobileNetV2作为基础模型
        self.model = models.mobilenet_v2(pretrained=True)
        
        # 冻结基础模型的参数
        for param in self.model.parameters():
            param.requires_grad = False
            
        # 修改最后的分类层
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)

def load_scene_dataset(data_root, scene_name):
    """加载特定场景的数据集"""
    scene_path = os.path.join(data_root, scene_name)
    if not os.path.isdir(scene_path):
        print(f"找不到场景目录: {scene_path}")
        return [], [], []
    
    image_paths = []
    labels = []
    class_names = []
    
    # 遍历场景下的所有类别
    categories = sorted(os.listdir(scene_path))
    for category in categories:
        category_path = os.path.join(scene_path, category)
        if not os.path.isdir(category_path):
            continue
        
        # 添加类别
        if category not in class_names:
            class_names.append(category)
        
        class_idx = class_names.index(category)
        
        # 遍历该类别下的所有图像
        for img_file in os.listdir(category_path):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(category_path, img_file)
                image_paths.append(img_path)
                labels.append(class_idx)
    
    print(f"场景 '{scene_name}' 加载了 {len(image_paths)} 个图像，{len(class_names)} 个类别: {class_names}")
    return image_paths, labels, class_names

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25):
    """训练模型并返回最佳模型"""
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # 创建结果记录
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    # 记录训练开始时间
    since = time.time()
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # 每个epoch都有训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置模型为训练模式
            else:
                model.eval()   # 设置模型为评估模式
            
            running_loss = 0.0
            running_corrects = 0
            
            # 迭代数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # 梯度清零
                optimizer.zero_grad()
                
                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # 反向传播 + 优化（仅在训练阶段）
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train' and scheduler is not None:
                scheduler.step()
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # 记录历史
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
            
            # 如果是最佳验证精度，保存模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        print()
    
    # 记录训练结束时间
    time_elapsed = time.time() - since
    print(f'训练完成，用时 {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'最佳验证集准确率: {best_acc:.4f}')
    
    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model, history

def fine_tune_model(model, dataloaders, num_epochs=10):
    """微调模型，解冻所有层并以较小的学习率训练"""
    # 解冻所有层
    for param in model.parameters():
        param.requires_grad = True
    
    # 使用较小的学习率
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    print("开始微调模型...")
    model, history = train_model(
        model, dataloaders, criterion, optimizer, scheduler, num_epochs=num_epochs
    )
    
    return model, history

def train_scene_model(data_root, scene_name, batch_size=16, num_epochs=20, fine_tune_epochs=10):
    """训练特定场景的模型"""
    print(f"\n{'='*50}")
    print(f"训练场景 '{scene_name}' 的模型")
    print(f"{'='*50}\n")
    
    # 加载场景数据集
    image_paths, labels, class_names = load_scene_dataset(data_root, scene_name)
    
    if not image_paths:
        print(f"场景 '{scene_name}' 没有有效数据，跳过训练")
        return None, None, None
    
    # 分割训练集和验证集
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # 创建数据集
    train_dataset = HandShadowDataset(train_paths, train_labels, transform=data_transforms['train'])
    val_dataset = HandShadowDataset(val_paths, val_labels, transform=data_transforms['val'])
    
    # 创建数据加载器
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    }
    
    # 创建模型
    model = HandShadowNet(num_classes=len(class_names))
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.model.classifier.parameters(), lr=0.001)
    
    # 定义学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # 训练模型
    print("第一阶段训练（只训练分类器层）...")
    model, history = train_model(
        model, dataloaders, criterion, optimizer, scheduler, num_epochs=num_epochs
    )
    
    # 微调模型
    model, fine_tune_history = fine_tune_model(model, dataloaders, num_epochs=fine_tune_epochs)
    
    # 合并历史记录
    for key in history:
        history[key].extend(fine_tune_history[key])
    
    # 保存模型和类别名称
    output_dir = "models"
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, f"{scene_name}_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names
    }, model_path)
    
    # 保存训练历史记录
    history_path = os.path.join(output_dir, f"{scene_name}_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f)
    
    print(f"模型已保存到 {model_path}")
    print(f"类别: {class_names}")
    
    # 绘制训练历史
    plot_training_history(history, scene_name, output_dir)
    
    return model, class_names, history

def plot_training_history(history, scene_name, output_dir):
    """绘制训练历史并保存图表"""
    plt.figure(figsize=(12, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.title(f'{scene_name} - 训练和验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='训练准确率')
    plt.plot(history['val_acc'], label='验证准确率')
    plt.title(f'{scene_name} - 训练和验证准确率')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(os.path.join(output_dir, f"{scene_name}_training_history.png"))
    plt.close()

def load_model(model_path):
    """加载训练好的模型"""
    checkpoint = torch.load(model_path)
    class_names = checkpoint['class_names']
    
    model = HandShadowNet(num_classes=len(class_names))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, class_names

def test_model_accuracy(data_root, scene_name, model, class_names):
    """测试模型在测试集上的准确率"""
    # 加载场景数据集
    image_paths, labels, _ = load_scene_dataset(data_root, scene_name)
    
    # 分割训练集和测试集
    _, test_paths, _, test_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # 创建测试数据集
    test_dataset = HandShadowDataset(test_paths, test_labels, transform=data_transforms['val'])
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    # 评估模型
    model.eval()
    correct = 0
    total = 0
    
    class_correct = list(0. for i in range(len(class_names)))
    class_total = list(0. for i in range(len(class_names)))
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 按类别统计
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    # 打印总体准确率
    print(f"\n{scene_name} 模型在测试集上的准确率: {100 * correct / total:.2f}%")
    
    # 打印每个类别的准确率
    for i in range(len(class_names)):
        print(f'类别 {class_names[i]} 的准确率: {100 * class_correct[i] / class_total[i]:.2f}%')
    
    return 100 * correct / total

def predict_hand_shadow(model, image, class_names, transform):
    """预测单个图像的类别"""
    # 预处理图像
    processed_img = HandShadowDataset.preprocess_hand_image(image)
    processed_img = Image.fromarray(processed_img)
    
    # 应用变换
    processed_img = transform(processed_img)
    processed_img = processed_img.unsqueeze(0)  # 添加批次维度
    processed_img = processed_img.to(device)
    
    # 进行预测
    with torch.no_grad():
        outputs = model(processed_img)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        _, predicted = torch.max(outputs, 1)
    
    # 获取预测结果
    class_idx = predicted.item()
    confidence = probabilities[class_idx].item()
    
    return {
        'class_name': class_names[class_idx],
        'confidence': confidence,
        'all_probabilities': {class_names[i]: probabilities[i].item() for i in range(len(class_names))}
    }

class ResultStabilizer:
    """结果稳定器，用于平滑预测结果"""
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.history = []
    
    def update(self, prediction):
        self.history.append(prediction)
        if len(self.history) > self.window_size:
            self.history.pop(0)
    
    def get_stable_result(self):
        if not self.history:
            return None
            
        # 统计各类别得票数
        votes = {}
        for pred in self.history:
            class_name = pred['class_name']
            votes[class_name] = votes.get(class_name, 0) + 1
        
        # 找到得票最多的类别
        max_votes = 0
        stable_class = None
        for class_name, count in votes.items():
            if count > max_votes:
                max_votes = count
                stable_class = class_name
        
        # 计算平均置信度
        avg_confidence = sum(pred['confidence'] for pred in self.history 
                           if pred['class_name'] == stable_class) / max_votes
        
        return {
            'class_name': stable_class,
            'confidence': avg_confidence,
            'vote_count': max_votes,
            'total_votes': len(self.history)
        }

def detect_hand(frame):
    """检测手部并提取轮廓"""
    # 转换到HSV颜色空间
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 创建皮肤掩码
    lower_skin = np.array([0, 48, 80], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv_frame, lower_skin, upper_skin)
    
    # 应用形态学操作
    kernel = np.ones((5, 5), np.uint8)
    skin_mask = cv2.erode(skin_mask, kernel, iterations=1)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
    
    # 提取轮廓
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    
    # 获取最大轮廓
    main_contour = max(contours, key=cv2.contourArea)
    
    # 检查轮廓面积是否足够大
    if cv2.contourArea(main_contour) < 1000:
        return None, None
    
    return skin_mask, main_contour

def real_time_recognition(scene_name=None):
    """实时手影识别系统"""
    models_dir = "models"
    
    # 加载所有场景的模型（或指定场景的模型）
    scene_models = {}
    scenes = []
    
    if scene_name:
        # 只加载指定场景的模型
        scenes = [scene_name]
    else:
        # 加载所有可用场景模型
        for file in os.listdir(models_dir):
            if file.endswith("_model.pth"):
                scene = file.split("_model.pth")[0]
                scenes.append(scene)
    
    for scene in scenes:
        model_path = os.path.join(models_dir, f"{scene}_model.pth")
        try:
            model, class_names = load_model(model_path)
            scene_models[scene] = {
                'model': model,
                'class_names': class_names
            }
            print(f"已加载场景 '{scene}' 的模型，类别: {class_names}")
        except Exception as e:
            print(f"加载 '{scene}' 模型失败: {e}")
    
    if not scene_models:
        print("没有找到可用的模型，请先训练模型")
        return
    
    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    
    # 初始化结果稳定器
    stabilizers = {scene: ResultStabilizer(window_size=8) for scene in scene_models}
    
    # 预处理变换
    transform = data_transforms['val']
    
    # 当前活跃场景
    active_scene = scenes[0] if scenes else None
    
    print(f"按 'q' 键退出，按数字键切换场景")
    print(f"当前场景: {active_scene}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 检测手部
        skin_mask, main_contour = detect_hand(frame)
        
        if main_contour is not None:
            # 获取轮廓的边界框
            x, y, w, h = cv2.boundingRect(main_contour)
            
            # 提取手部区域（稍微扩大边界框）
            padding = 20
            x1, y1 = max(0, x - padding), max(0, y - padding)
            x2, y2 = min(frame.shape[1], x + w + padding), min(frame.shape[0], y + h + padding)
            hand_img = frame[y1:y2, x1:x2].copy()
            
            if hand_img.size > 0 and active_scene:  # 确保提取的区域有效
                # 预测手影类别
                model_info = scene_models[active_scene]
                prediction = predict_hand_shadow(model_info['model'], hand_img, model_info['class_names'], transform)
                
                # 更新稳定器
                stabilizers[active_scene].update(prediction)
                
                # 获取稳定结果
                stable_result = stabilizers[active_scene].get_stable_result()
                
                # 在帧上绘制结果
                cv2.drawContours(frame, [main_contour], -1, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                if stable_result:
                    cv2.putText(
                        frame,
                        f"Class: {stable_result['class_name']}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
                    
                    cv2.putText(
                        frame,
                        f"Confidence: {stable_result['confidence']:.2f}",
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
                    
                    cv2.putText(
                        frame,
                        f"Votes: {stable_result['vote_count']}/{stable_result['total_votes']}",
                        (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
        
        # 显示当前场景
        cv2.putText(
            frame,
            f"Scene: {active_scene}",
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        # 显示结果
        cv2.imshow("Hand Shadow Recognition", frame)
        
        # 按键处理
        key = cv2.waitKey(1) & 0xFF
        
        # 按q退出
        if key == ord('q'):
            break
            
        # 数字键切换场景
        for i, scene in enumerate(scenes):
            if key == ord(str(i + 1)):
                active_scene = scene
                print(f"切换到场景: {active_scene}")
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

def main(data_root):
    """主函数，训练所有场景的模型"""
    # 获取所有场景
    scenes = []
    for item in os.listdir(data_root):
        scene_path = os.path.join(data_root, item)
        if os.path.isdir(scene_path):
            scenes.append(item)
    
    print(f"发现以下场景: {scenes}")
    
    # 训练每个场景的模型
    scene_results = {}
    for scene in scenes:
        model, class_names, history = train_scene_model(
            data_root=data_root,
            scene_name=scene,
            batch_size=16,
            num_epochs=15,  # 第一阶段训练轮数
            fine_tune_epochs=10  # 微调轮数
        )
        
        if model and class_names:
            # 测试模型准确率
            accuracy = test_model_accuracy(data_root, scene, model, class_names)
            scene_results[scene] = {
                "class_names": class_names,
                "accuracy": accuracy
            }
    
    # 打印总结
    print("\n" + "="*50)
    print("训练完成! 模型准确率总结:")
    print("="*50)
    for scene, result in scene_results.items():
        print(f"场景 '{scene}': {result['accuracy']:.2f}%, 类别: {result['class_names']}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="手影识别训练和预测")
    parser.add_argument("--data_root", type=str, help="数据集根目录")
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train", help="模式: train或test")
    parser.add_argument("--scene", type=str, help="指定场景(仅测试模式使用)")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        if not args.data_root:
            print("训练模式需要指定数据集根目录，使用 --data_root 参数")
        else:
            main(args.data_root)
    else:  # test mode
        real_time_recognition(args.scene)