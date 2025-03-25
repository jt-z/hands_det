import cv2

# 打开视频文件
input_video = 'output_dog_video_final.mp4'
output_video = '/Users/zjt/Documents/Develop/AI_CV_Projects/Dir_PoseEstimation/HandPoseShadow/output_dog_video_final_processed.mp4'


# 读取视频
cap = cv2.VideoCapture(input_video)
cap = cv2.VideoCapture(input_video)

# 检查视频是否正确打开
if not cap.isOpened():
    print("无法打开视频文件")
    exit()

# 获取视频的宽度、高度和帧率
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 定义视频输出格式
fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # 使用MJPEG编码

out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

# 循环处理每一帧
while True:
    ret, frame = cap.read()

    if not ret:
        break

    # 裁剪左下角的500x300像素区域
    cropped_region = frame[frame_height-300:frame_height, 0:500]

    # 将裁剪的区域复制到左上角
    frame[0:300, 0:500] = cropped_region

    # 写入处理后的视频帧
    out.write(frame)

# 释放视频流
cap.release()
out.release()

print("处理完成，视频已保存为:", output_video)
