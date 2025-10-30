import cv2
import subprocess

# --- 配置 ---
# 摄像头索引
CAMERA_INDEX = 0 
# RTSP 服务器推流地址
# "mystream" 是你可以自定义的流名称
RTSP_URL = 'rtsp://127.0.0.1:8554/mystream'

# --- 1. 使用 OpenCV 打开摄像头 ---
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("错误: 无法打开摄像头")
    exit()

# 获取摄像头的属性
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 确保获取到了有效的 FPS
if fps <= 0:
    print("警告: 无法获取摄像头FPS, 设为默认值 30")
    fps = 30 # 设置一个合理的默认值

print(f"摄像头分辨率: {width}x{height}, FPS: {fps}")

# --- 2. 配置 FFmpeg 子进程 ---
# FFmpeg 命令
# -f rawvideo: 输入格式为原始视频
# -vcodec rawvideo: 输入编解码器
# -pix_fmt bgr24: OpenCV 默认的像素格式
# -s WxH: 输入分辨率
# -r FPS: 输入帧率
# -i -: 从标准输入 (stdin) 读取
# -c:v libx264: 输出视频编码为 H.264
# -pix_fmt yuv420p: H.264 兼容的像素格式
# -preset ultrafast: 快速编码，低延迟
# -f rtsp: 输出格式为 RTSP
# RTSP_URL: 推流地址
command = [
    'ffmpeg',
    '-y',                 # 覆盖输出文件（在这里主要用于rtsp）
    '-f', 'rawvideo',
    '-vcodec', 'rawvideo',
    '-pix_fmt', 'bgr24',
    '-s', f'{width}x{height}',
    '-r', str(fps),
    '-i', '-',            # 从 stdin 读取输入
    '-c:v', 'libx264',
    '-pix_fmt', 'yuv420p',
    '-preset', 'ultrafast',
    '-f', 'rtsp',
    RTSP_URL
]

# 启动 FFmpeg 子进程
# stdin=subprocess.PIPE 允许我们向 FFmpeg 写入数据
try:
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
except FileNotFoundError:
    print("错误: FFmpeg 未找到。请确保它已安装并在系统 PATH 中。")
    cap.release()
    exit()

print("开始推流... 按 'q' 键停止。")

# --- 3. 循环读取帧并推送 ---
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("错误: 无法读取摄像头帧")
            break

        # (可选) 在这里可以对 frame 进行处理，例如
        # cv2.putText(frame, "LIVE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # 将帧数据写入 FFmpeg 的 stdin
        # 必须将 frame 转换为原始字节
        try:
            process.stdin.write(frame.tobytes())
        except IOError as e:
            # 捕获管道破裂等错误 (例如 FFmpeg 进程意外退出)
            print(f"FFmpeg 进程错误: {e}")
            break

        # (可选) 显示本地预览窗口
        # cv2.imshow('Local Preview', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

except KeyboardInterrupt:
    # 捕获 Ctrl+C
    print("停止推流...")
finally:
    # --- 4. 清理 ---
    print("清理资源...")
    cap.release()
    # cv2.destroyAllWindows() # 如果你用了 imshow
    
    # 关闭 FFmpeg 进程
    if 'process' in locals() and process.stdin:
        process.stdin.close()
    if 'process' in locals():
        process.wait()
    print("推流已停止。")