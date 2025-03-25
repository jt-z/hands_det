# fps is low : about 3fps , but is ok. 
# seg result is ok.
import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 加载预训练的DeepLabV3模型
model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
print('evaling:')
model.eval()

# 定义图像预处理操作
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((520, 520)),  # 调整为网络要求的大小
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 定义后处理操作，将输出转换回图像
def postprocess(output):
    output_predictions = output['out'].argmax(1).squeeze(0).cpu().numpy()
    return output_predictions

# 打开摄像头或视频文件
cap = cv2.VideoCapture('dog.mov')
cap = cv2.VideoCapture(0)

# 获取视频的帧率和尺寸
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 定义视频保存对象
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 使用XVID编码
out = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))

cnt = 0
while cap.isOpened():
    ret, frame = cap.read()
    cnt += 1
    if cnt < 300:
        continue
    if not ret:
        print("Failed to capture video frame.")
        break

    # 转换为RGB格式
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 预处理图像
    input_tensor = preprocess(frame_rgb)
    input_batch = input_tensor.unsqueeze(0)

    # 确保模型在没有梯度的情况下运行，以加速计算
    with torch.no_grad():
        output = model(input_batch)

    # 后处理得到分割掩码
    segmentation_mask = postprocess(output)

    # 处理分割结果，将其转换为可视化格式
    seg_image = np.where(segmentation_mask[..., None] == 15, 255, 0).astype(np.uint8)  # 人体类ID为15
    seg_image = cv2.cvtColor(seg_image, cv2.COLOR_GRAY2BGR)

    # 打印frame和seg_image的形状，调试错误
    print(f"Frame shape: {frame.shape}")
    print(f"Segmentation image shape: {seg_image.shape}")

    # 如果frame和seg_image大小不同，调整seg_image的大小
    if frame.shape[:2] != seg_image.shape[:2]:
        seg_image = cv2.resize(seg_image, (frame.shape[1], frame.shape[0]))

    # 将原始图像与分割掩码叠加显示
    combined_image = cv2.addWeighted(frame, 0.5, seg_image, 0.5, 0)

    # 保存到视频
    out.write(combined_image)

    # 显示结果
    cv2.imshow('Real-time Image Segmentation', combined_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头和视频写入对象并关闭窗口
cap.release()
out.release()
cv2.destroyAllWindows()
