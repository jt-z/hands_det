#!/usr/bin/env python3
"""
手影识别系统启动脚本
提供便捷的启动选项
"""

import os
import sys
import argparse

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='手影识别系统启动脚本')
    
    # 运行模式选择
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--camera', action='store_true', help='使用摄像头模式')
    mode_group.add_argument('--video', type=str, help='使用视频文件模式')
    mode_group.add_argument('--server-only', action='store_true', help='仅启动命令服务器')
    
    # 手影组选择
    parser.add_argument('--group', type=str, choices=['group1', 'group2', 'group3', 'group4', 'group5'],
                       help='初始手影组')
    
    # 网络设置
    parser.add_argument('--send-ip', type=str, help='结果发送目标IP')
    parser.add_argument('--send-port', type=int, help='结果发送目标端口')
    parser.add_argument('--listen-ip', type=str, help='命令监听IP')
    parser.add_argument('--listen-port', type=int, help='命令监听端口')
    
    # 其他选项
    parser.add_argument('--skip', type=int, help='帧跳过数')
    parser.add_argument('--show', action='store_true', help='显示预览窗口')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 构建命令行参数
    cmd_args = []
    
    # 设置运行模式
    if args.camera:
        cmd_args.append(f"--camera=0")
    elif args.video:
        cmd_args.append(f"--video={args.video}")
    elif args.server_only:
        # 仅启动服务器模式不需要额外参数
        pass
        
    # 添加其他参数
    if args.group:
        cmd_args.append(f"--group={args.group}")
    if args.send_ip:
        cmd_args.append(f"--send-ip={args.send_ip}")
    if args.send_port:
        cmd_args.append(f"--send-port={args.send_port}")
    if args.listen_ip:
        cmd_args.append(f"--listen-ip={args.listen_ip}")
    if args.listen_port:
        cmd_args.append(f"--listen-port={args.listen_port}")
    if args.skip:
        cmd_args.append(f"--skip={args.skip}")
    if args.show:
        cmd_args.append("--show")
    if args.debug:
        cmd_args.append("--debug")
    
    # 构建完整命令
    cmd = [sys.executable, "handpose_shadow/main.py"] + cmd_args
    cmd_str = " ".join(cmd)
    
    print(f"启动命令: {cmd_str}")
    
    # 执行命令
    os.execv(cmd[0], cmd)

if __name__ == "__main__":
    main()