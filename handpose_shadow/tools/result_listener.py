"""
结果监听工具
用于监听和显示手影识别系统发送的识别结果
"""

import os
import sys
import socket
import json
import argparse
import threading
import time
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from handpose_shadow.config import UDP_SEND_IP, UDP_SEND_PORT

class ResultListener:
    """结果监听器类，用于接收和显示识别结果"""
    
    def __init__(self, listen_ip, listen_port):
        """
        初始化结果监听器
        
        参数:
            listen_ip (str): 监听IP地址
            listen_port (int): 监听端口
        """
        self.listen_ip = listen_ip
        self.listen_port = listen_port
        self.socket = None
        self.running = False
        self.thread = None
        self.received_count = 0
    
    def start(self):
        """启动监听"""
        if self.running:
            print("监听器已在运行")
            return
        
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.listen_ip, self.listen_port))
            self.socket.settimeout(0.5)
            
            self.running = True
            self.thread = threading.Thread(target=self._listen_loop)
            self.thread.daemon = True
            self.thread.start()
            
            print(f"结果监听器已启动，监听: {self.listen_ip}:{self.listen_port}")
            
        except Exception as e:
            print(f"启动监听器失败: {e}")
    
    def stop(self):
        """停止监听"""
        if not self.running:
            print("监听器未在运行")
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            
        if self.socket:
            self.socket.close()
            self.socket = None
            
        print("结果监听器已停止")
    
    def _listen_loop(self):
        """监听循环"""
        print("等待接收识别结果...")
        
        while self.running:
            try:
                data, addr = self.socket.recvfrom(1024)
                if data:
                    self._process_result(data, addr)
                    
            except socket.timeout:
                # 超时，继续循环
                continue
                
            except Exception as e:
                if self.running:  # 只有在正常运行时才记录错误
                    print(f"监听循环出错: {e}")
    
    def _process_result(self, data, addr):
        """
        处理接收到的结果
        
        参数:
            data (bytes): 接收到的数据
            addr (tuple): 发送方地址
        """
        try:
            # 解析JSON数据
            result = json.loads(data.decode('utf-8'))
            
            # 增加接收计数
            self.received_count += 1
            
            # 获取时间戳
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            
            # 打印基本信息
            print(f"\n[{timestamp}] 收到结果 #{self.received_count} 来自: {addr[0]}:{addr[1]}")
            
            # 判断结果类型
            if "id" in result and "name" in result:
                # 识别结果
                print(f"识别结果: {result['name']} (ID: {result['id']})")
                print(f"相似度: {result.get('similarity', 'N/A')}")
                print(f"分组: {result.get('group', 'N/A')}")
                print(f"时间戳: {result.get('timestamp', 'N/A')}")
                
            elif "status" in result:
                # 状态消息
                status = result["status"]
                message = result.get("message", "")
                status_str = "成功" if status == "ok" else "错误"
                print(f"状态消息: {status_str}")
                print(f"内容: {message}")
                
            else:
                # 未知结果类型
                print("未知结果类型:")
                print(json.dumps(result, ensure_ascii=False, indent=2))
            
        except json.JSONDecodeError:
            print(f"无效的JSON数据: {data}")
            
        except Exception as e:
            print(f"处理结果时出错: {e}")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='手影识别系统结果监听工具')
    
    parser.add_argument('--ip', type=str, default='0.0.0.0',
                      help='监听IP地址，默认为0.0.0.0（所有接口）')
    parser.add_argument('--port', type=int, default=UDP_SEND_PORT,
                      help=f'监听端口，默认为{UDP_SEND_PORT}')
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 创建结果监听器
    listener = ResultListener(args.ip, args.port)
    
    try:
        # 启动监听
        listener.start()
        
        # 主线程等待
        print("监听器已启动，按Ctrl+C退出...")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n收到中断信号，退出中...")
        
    finally:
        # 停止监听
        listener.stop()

if __name__ == "__main__":
    main()
