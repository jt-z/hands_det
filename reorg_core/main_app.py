import cv2 as cv
import time
import numpy as np
from hand_utils import (
    skinmask, 
    getcnthull, 
    overlay_foreground_on_left, 
    compare_contours,
    prepare_template_mask
)
from network_utils import send_score_to_ip

# 配置参数
class Config:
    def __init__(self):
        self.template_path = 'mask_dog526w512h.png'  # 模板图像路径
        self.template_path = 'D:\Documents\Onedrive\Documents\A_Dashboard\PartTime\HandPoseShadow\\assets\mask_dog.png'  # 模板图像路径
        # D:\Documents\Onedrive\Documents\A_Dashboard\PartTime\HandPoseShadow\assets\mask_dog.png
        self.similarity_threshold = 55  # 相似度阈值
        self.consecutive_frames_threshold = 3  # 连续成功帧数阈值
        self.frame_skip = 5  # 每隔多少帧处理一次
        self.ip_address = "127.0.0.1"  # UDP目标IP
        self.udp_port = 9890  # UDP目标端口
        self.dog_id = '1002'  # 发送的ID信息
        
        # 视频源设置
        self.use_camera = False  # 是否使用摄像头
        self.camera_id = 0  # 摄像头ID，通常0是默认摄像头
        self.video_path = ' D:\Documents\Onedrive\Documents\A_Dashboard\PartTime\HandPoseShadow\core_methods\dl_seg\dog.mov'  # 本地视频文件路径
        self.video_source = self.camera_id if self.use_camera else self.video_path  # 根据设置决定视频源
        
        self.save_output = True  # 是否保存输出视频
        self.output_filename = None  # 自动根据源视频生成输出文件名

def main():
    # 加载配置
    config = Config()
    
    # 打开摄像头或视频文件
    print(f"{'使用摄像头' if config.use_camera else '打开视频文件'}: {config.video_source}")
    cap = cv.VideoCapture(config.video_source)
    if not cap.isOpened():
        print(f"无法打开视频源: {config.video_source}")
        return

    # 获取视频属性
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv.CAP_PROP_FPS))
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT)) if not config.use_camera else 0
    
    # 打印视频信息
    print(f"视频分辨率: {frame_width}x{frame_height}")
    print(f"视频帧率: {fps}")
    if total_frames > 0:
        print(f"视频总帧数: {total_frames}")
        print(f"视频时长: {total_frames/fps:.2f}秒")

    # 准备模板
    dog_mask_resized, template_contours = prepare_template_mask(
        config.template_path, frame_width, frame_height
    )
    
    # 设置视频写入器
    out = None
    if config.save_output:
        # 自动生成输出文件名
        if config.output_filename is None:
            if config.use_camera:
                output_filename = f'output_camera_{time.strftime("%Y%m%d_%H%M%S")}.avi'
            else:
                # 从原视频路径提取文件名
                import os
                base_name = os.path.basename(config.video_path)
                name_without_ext = os.path.splitext(base_name)[0]
                output_filename = f'output_{name_without_ext}.avi'
        else:
            output_filename = config.output_filename
            
        print(f"输出视频将保存为: {output_filename}")
        out = cv.VideoWriter(
            output_filename, 
            cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 
            fps, 
            (frame_width * 2, frame_height)
        )

    # 初始化计数器
    cnt_frames = 0
    cnt_score_ok = 0
    
    print(f"开始处理视频，每{config.frame_skip}帧处理一次")
    
    # 主循环
    start_time = time.time()
    while cap.isOpened():
        ret, img = cap.read()
        cnt_frames += 1
        
        # 显示进度信息（仅对视频文件）
        if not config.use_camera and total_frames > 0 and cnt_frames % 30 == 0:
            progress = (cnt_frames / total_frames) * 100
            elapsed_time = time.time() - start_time
            estimated_total = elapsed_time / (cnt_frames / total_frames)
            remaining_time = estimated_total - elapsed_time
            print(f"进度: {progress:.1f}% ({cnt_frames}/{total_frames}), " +
                  f"已用时间: {elapsed_time:.1f}秒, 剩余时间: {remaining_time:.1f}秒")
        
        # 帧率控制
        if cnt_frames % config.frame_skip != 0:
            if cnt_frames % 30 == 0 and config.use_camera:
                print(f'每{config.frame_skip}帧检测一次,每30帧打印一次此信息')
            continue

        if not ret:
            print("视频读取结束")
            break
            
        # 检查图像是否为空
        if img is None or img.size == 0:
            print("警告: 检测到空帧，跳过处理")
            continue

        try:
            # 生成皮肤掩码并提取轮廓
            mask_img = skinmask(img)
            contours, hull = getcnthull(mask_img)
            if contours is None:
                continue

            # 计算相似度
            similarity_score = compare_contours(template_contours, contours)

            # 绘制手掌轮廓
            cv.drawContours(img, [contours], -1, (255, 255, 0), 2)
            
            # 创建结果图像
            result_frame = overlay_foreground_on_left(img, mask_img, dog_mask_resized)

            # 显示相似度分数
            cv.putText(
                result_frame, 
                f"Similarity: {similarity_score:.2f}", 
                (50, 100), 
                cv.FONT_HERSHEY_SIMPLEX, 
                1.5, 
                (255, 122, 125), 
                3, 
                cv.LINE_AA
            )

            # 检查相似度是否超过阈值
            if similarity_score > config.similarity_threshold:
                cnt_score_ok += 1
                if cnt_score_ok > config.consecutive_frames_threshold:
                    cv.putText(
                        result_frame, 
                        "Success", 
                        (50, 200), 
                        cv.FONT_HERSHEY_SIMPLEX, 
                        1.5, 
                        (0, 255, 0), 
                        3, 
                        cv.LINE_AA
                    )
                    # 发送UDP消息
                    print(f'发送消息到UDP服务器: {config.ip_address}')
                    send_score_to_ip(
                        config.dog_id, 
                        config.ip_address, 
                        config.udp_port
                    )
                    cnt_score_ok = 0
            else:
                cv.putText(
                    result_frame, 
                    "Not ok yet.", 
                    (50, 200), 
                    cv.FONT_HERSHEY_SIMPLEX, 
                    1.5, 
                    (0, 0, 255), 
                    3, 
                    cv.LINE_AA
                )

            # 控制处理速度
            time.sleep(0.05)

            # 显示结果
            cv.imshow("Hand Tracking", result_frame)
            
            # 保存输出视频
            if config.save_output and out is not None:
                out.write(result_frame)

        except Exception as e:
            print(f"错误: {e}")
            pass

        # 按q退出
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    if out is not None:
        out.release()
    cv.destroyAllWindows()
    print("程序结束")

def parse_arguments():
    """解析命令行参数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='手部姿势检测与模板匹配')
    
    # 视频源相关参数
    video_group = parser.add_mutually_exclusive_group()
    video_group.add_argument('--camera', default = True ,action='store_true', help='使用摄像头')
    video_group.add_argument('--video', type=str, help='指定视频文件路径')
    parser.add_argument('--camera-id', type=int, default=0, help='摄像头ID，默认为0')
    
    # 处理参数
    parser.add_argument('--skip', type=int, help='跳帧数，每隔多少帧处理一次')
    parser.add_argument('--threshold', type=float, help='相似度阈值')
    parser.add_argument('--template', type=str, help='模板图像路径')
    
    # 输出相关参数
    parser.add_argument('--no-save', action='store_true', help='不保存输出视频')
    parser.add_argument('--output', type=str, help='输出视频文件名')
    
    # 网络相关参数
    parser.add_argument('--ip', type=str, help='UDP目标IP地址')
    parser.add_argument('--port', type=int, help='UDP目标端口')
    parser.add_argument('--id', type=str, help='发送的ID信息')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # 解析命令行参数
    args = parse_arguments()
    
    # 如果提供了命令行参数，则应用到配置
    if any(vars(args).values()):
        config = Config()
        
        # 应用视频源设置
        if args.camera:
            config.use_camera = True
            if args.camera_id is not None:
                config.camera_id = args.camera_id
            config.video_source = config.camera_id
        elif args.video:
            config.use_camera = False
            config.video_path = args.video
            config.video_source = config.video_path
        
        # 应用其他参数
        if args.skip:
            config.frame_skip = args.skip
        if args.threshold:
            config.similarity_threshold = args.threshold
        if args.template:
            config.template_path = args.template
        if args.no_save:
            config.save_output = False
        if args.output:
            config.output_filename = args.output
        if args.ip:
            config.ip_address = args.ip
        if args.port:
            config.udp_port = args.port
        if args.id:
            config.dog_id = args.id
            
        # 打印配置信息
        print("使用以下配置:")
        for key, value in vars(config).items():
            print(f"  {key}: {value}")
    
    # 运行主程序
    main()