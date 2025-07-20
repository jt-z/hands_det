"""
命令客户端工具
用于测试向手影识别系统发送命令
"""

import os
import sys
import socket
import json
import argparse
import time
from datetime import datetime
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from handpose_shadow.config import COMMAND_TYPES, UDP_LISTEN_IP, UDP_LISTEN_PORT

class CommandClient:
    """命令客户端类，用于发送命令和接收响应"""
    
    def __init__(self, server_ip, server_port, timeout=2.0):
        """
        初始化命令客户端
        
        参数:
            server_ip (str): 服务器IP地址
            server_port (int): 服务器端口
            timeout (float): 超时时间（秒）
        """
        self.server_ip = server_ip
        self.server_port = server_port
        self.timeout = timeout
        self.socket = None
    
    def connect(self):
        """连接到服务器"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.settimeout(self.timeout)
        print(f"命令客户端已初始化，目标: {self.server_ip}:{self.server_port}")
    
    def close(self):
        """关闭连接"""
        if self.socket:
            self.socket.close()
            self.socket = None
    
    def send_command(self, command):
        """
        发送命令
        
        参数:
            command (dict): 命令数据
            
        返回:
            dict: 服务器响应，如果超时则返回None
        """
        if not self.socket:
            self.connect()
        
        try:
            # 将命令转换为JSON并编码
            data = json.dumps(command).encode('utf-8')
            
            # 添加时间戳
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            
            # 发送命令
            print(f"[{timestamp}] 发送命令: {command}")
            self.socket.sendto(data, (self.server_ip, self.server_port))
            
            # 接收响应
            response_data, addr = self.socket.recvfrom(1024)
            response = json.loads(response_data.decode('utf-8'))
            
            print(f"[{timestamp}] 收到响应: {response} 来自: {addr[0]}:{addr[1]}")
            
            return response
            
        except socket.timeout:
            print(f"[{timestamp}] 等待响应超时")
            return None
            
        except Exception as e:
            print(f"[{timestamp}] 发送命令出错: {e}")
            return None

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='手影识别系统命令客户端')
    
    # 服务器设置
    parser.add_argument('--ip', type=str, default='127.0.0.1',
                      help='服务器IP地址')
    parser.add_argument('--port', type=int, default=UDP_LISTEN_PORT,
                      help='服务器端口')
    
    # 命令选择
    command_group = parser.add_mutually_exclusive_group(required=True)
    command_group.add_argument('--start', type=str, nargs='?', const='',
                             help='发送开始命令，可选择指定场景ID')
    command_group.add_argument('--stop', type=str, nargs='?', const='',
                             help='发送停止命令，可选择指定场景ID')
    command_group.add_argument('--switch', type=str,
                             help='发送切换场景命令，需要指定场景ID')
    command_group.add_argument('--ping', action='store_true',
                             help='发送心跳检测命令')
    command_group.add_argument('--left', type=str,
                             help='发送左侧命令，需要指定场景ID')
    command_group.add_argument('--right', type=str,
                             help='发送右侧命令，需要指定场景ID')
    command_group.add_argument('--interactive', action='store_true',
                             help='进入交互模式')
    
    return parser.parse_args()

def show_interactive_menu():
    """显示交互式菜单"""
    print("\n==== 手影识别系统命令菜单 ====")
    print("1. 开始识别")
    print("2. 停止识别")
    print("3. 切换场景")
    print("4. 发送心跳检测")
    print("5. 发送左侧命令")
    print("6. 发送右侧命令")
    print("0. 退出")
    print("============================")
    return input("请选择操作: ")

def interactive_mode(client):
    """交互模式"""
    while True:
        choice = show_interactive_menu()
        
        if choice == '0':
            print("退出交互模式")
            break
            
        elif choice == '1':
            scene_id = input("请输入场景ID (直接回车使用默认值): ")
            command = {"type": COMMAND_TYPES["START"]}
            if scene_id:
                command["scene_id"] = scene_id
            client.send_command(command)
            
        elif choice == '2':
            scene_id = input("请输入场景ID (直接回车使用默认值): ")
            command = {"type": COMMAND_TYPES["STOP"]}
            if scene_id:
                command["scene_id"] = scene_id
            client.send_command(command)
            
        elif choice == '3':
            scene_id = input("请输入场景ID: ")
            if scene_id:
                client.send_command({"type": COMMAND_TYPES["SWITCH_SCENE"], "scene_id": scene_id})
            else:
                print("场景ID不能为空")
                
        elif choice == '4':
            client.send_command({"type": COMMAND_TYPES["PING"]})
            
        elif choice == '5':
            scene_id = input("请输入场景ID: ")
            if scene_id:
                client.send_command({"type": COMMAND_TYPES["LEFT"], "scene_id": scene_id})
            else:
                print("场景ID不能为空")
                
        elif choice == '6':
            scene_id = input("请输入场景ID: ")
            if scene_id:
                client.send_command({"type": COMMAND_TYPES["RIGHT"], "scene_id": scene_id})
            else:
                print("场景ID不能为空")
                
        else:
            print("无效的选择")
        
        # 暂停一下，让用户看到响应
        time.sleep(1)

def main():
    """主函数"""
    args = parse_args()
    
    # 创建命令客户端
    client = CommandClient(args.ip, args.port)
    
    try:
        if args.interactive:
            interactive_mode(client)
            
        else:
            # 根据参数发送相应命令
            if args.start is not None:
                command = {"type": COMMAND_TYPES["START"]}
                if args.start:  # 如果提供了场景ID
                    command["scene_id"] = args.start
                client.send_command(command)
                
            elif args.stop is not None:
                command = {"type": COMMAND_TYPES["STOP"]}
                if args.stop:  # 如果提供了场景ID
                    command["scene_id"] = args.stop
                client.send_command(command)
                
            elif args.switch:
                client.send_command({
                    "type": COMMAND_TYPES["SWITCH_SCENE"],
                    "scene_id": args.switch
                })
                
            elif args.left:
                client.send_command({
                    "type": COMMAND_TYPES["LEFT"],
                    "scene_id": args.left
                })
                
            elif args.right:
                client.send_command({
                    "type": COMMAND_TYPES["RIGHT"],
                    "scene_id": args.right
                })
                
            elif args.ping:
                client.send_command({"type": COMMAND_TYPES["PING"]})
    
    finally:
        client.close()

if __name__ == "__main__":
    main()