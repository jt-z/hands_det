import openai
import cv2
import os

# 初始化 GPT API
openai.api_key = 'your-api-key'

def analyze_frame(frame):
    """
    使用 GPT 模型分析单个图像帧，返回手影类型和相似度评分
    """
    # 保存图片文件
    frame_path = "frame.jpg"
    cv2.imwrite(frame_path, frame)
    
    # 打开图片并进行上传
    with open(frame_path, 'rb') as image_file:
        # 调用 GPT 模型接口
        response = openai.Image.create(
            file=image_file,
            prompt="Analyze this shadow puppet image. Identify the shadow puppet and give a similarity score (0-100).",
            size="1024x1024",
            n=1
        )
        
        # 从返回结果中提取手影类型和相似度分数
        description = response['data'][0]['description']  # 假设返回的描述包含手影类型和评分
        return description

def process_video(video_path):
    """
    处理视频，将其拆解为帧并逐帧分析
    """
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    results = []
    
    # 遍历所有帧
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        
        # 调用 GPT 接口分析每一帧
        result = analyze_frame(frame)
        results.append(result)
    
    cap.release()
    
    return results

if __name__ == "__main__":
    video_file = '.mp4'  # 指定要上传的手影视频
    results = process_video(video_file)
    
    # 打印每一帧的分析结果
    for i, result in enumerate(results):
        print(f"Frame {i}: {result}")
