import cv2

def read_network_camera(stream_url):
    """
    使用 OpenCV 从网络摄像头 (RTSP 或 RTMP) 读取视频流并显示。

    参数:
    stream_url (str): 摄像头的完整 RTSP 或 RTMP URL。
    """
    
    # 尝试连接到视频流
    # cv2.VideoCapture() 既支持 RTSP 也支持 RTMP
    cap = cv2.VideoCapture(stream_url)

    # 检查连接是否成功
    if not cap.isOpened():
        print(f"错误: 无法打开视频流。")
        print(f"请检查 URL 是否正确: {stream_url}")
        print("可能的原因：")
        print("1. URL 拼写错误 (包括用户名、密码、IP地址、端口)。")
        print("2. 摄像头与本机网络不通 (检查 ping)。")
        print("3. 摄像头未开启 RTSP/RTMP 服务。")
        print("4. 防火墙阻止了连接。")
        return

    print("成功连接到视频流。按 'q' 键退出。")

    while True:
        # 逐帧读取
        # ret 是一个布尔值, 表示是否成功读取到帧
        # frame 是读取到的图像帧 (numpy 数组)
        ret, frame = cap.read()

        # 如果 ret 为 False，表示视频流结束或读取失败
        if not ret:
            print("错误: 无法读取帧，可能视频流已断开。")
            break

        # 在窗口中显示图像帧
        cv2.imshow('Network Camera Stream', frame)

        # 等待按键，如果按下 'q' 键 (ASCII 码) 则退出循环
        # cv2.waitKey(1) 表示等待 1 毫秒
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("正在退出...")
            break

    # 释放资源
    cap.release()
    # 关闭所有 OpenCV 创建的窗口
    cv2.destroyAllWindows()

# --- 主程序 ---
if __name__ == "__main__":
    
    # ####################################################################
    # ## 替换成你自己的 URL ##
    # ####################################################################
    
    # 示例 RTSP URL (你需要替换成你自己的)
    # 格式通常是: rtsp://[用户名]:[密码]@[IP地址]:[端口号]/[流路径]
    # rtsp_url = "rtsp://your_username:your_password@your_camera_ip:554/your_stream_path"
    rtsp_url = "rtsp://127.0.0.1:8554/mystream"

    # 示例 RTMP URL (你需要替换成你自己的)
    # rtmp_url = "rtmp://your_server_ip/live/your_stream_key"

    # 调用函数开始读取
    # 你需要把下面的 rtsp_url 换成你摄像头的实际 URL
    read_network_camera(rtsp_url) 