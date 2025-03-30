"""
系统集成测试工具
用于测试手影识别系统的完整功能
"""

import os
import sys
import argparse
import time
import threading
import subprocess
import signal

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 命令客户端和结果监听器工具路径
COMMAND_CLIENT_PATH = os.path.join(os.path.dirname(__file__), 'command_client.py')
RESULT_LISTENER_PATH = os.path.join(os.path.dirname(__file__), 'result_listener.py')

class SystemTest:
    """系统集成测试类"""
    
    def __init__(self, system_cmd, timeout=60):
        """
        初始化系统测试
        
        参数:
            system_cmd (str): 启动系统的命令
            timeout (int): 最大测试时间（秒）
        """
        self.system_cmd = system_cmd
        self.timeout = timeout
        self.system_process = None
        self.listener_process = None
        self.test_thread = None
        self.running = False
    
    def start_system(self):
        """启动系统进程"""
        try:
            print(f"启动系统: {self.system_cmd}")
            self.system_process = subprocess.Popen(
                self.system_cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # 创建线程读取和打印系统输出
            def print_output():
                for line in self.system_process.stdout:
                    print(f"[系统] {line.strip()}")
            
            threading.Thread(target=print_output, daemon=True).start()
            
            # 等待系统启动
            print("等待系统启动...")
            time.sleep(3)
            
            return True
            
        except Exception as e:
            print(f"启动系统失败: {e}")
            return False
    
    def start_result_listener(self):
        """启动结果监听进程"""
        try:
            cmd = f"{sys.executable} {RESULT_LISTENER_PATH}"
            print(f"启动结果监听器: {cmd}")
            
            self.listener_process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # 创建线程读取和打印监听器输出
            def print_output():
                for line in self.listener_process.stdout:
                    print(f"[监听器] {line.strip()}")
            
            threading.Thread(target=print_output, daemon=True).start()
            
            # 等待监听器启动
            print("等待结果监听器启动...")
            time.sleep(2)
            
            return True
            
        except Exception as e:
            print(f"启动结果监听器失败: {e}")
            return False
    
    def send_command(self, command_args):
        """
        发送命令
        
        参数:
            command_args (str): 命令参数
        
        返回:
            bool: 发送是否成功
        """
        try:
            cmd = f"{sys.executable} {COMMAND_CLIENT_PATH} {command_args}"
            print(f"发送命令: {cmd}")
            
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            
            # 等待命令完成并打印输出
            output, _ = process.communicate()
            for line in output.splitlines():
                print(f"[命令] {line.strip()}")
            
            return process.returncode == 0
            
        except Exception as e:
            print(f"发送命令失败: {e}")
            return False
    
    def stop_processes(self):
        """停止所有进程"""
        # 停止系统进程
        if self.system_process:
            print("停止系统进程...")
            try:
                self.system_process.terminate()
                self.system_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.system_process.kill()
            self.system_process = None
        
        # 停止监听器进程
        if self.listener_process:
            print("停止监听器进程...")
            try:
                self.listener_process.terminate()
                self.listener_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.listener_process.kill()
            self.listener_process = None
    
    def run_test_sequence(self):
        """运行测试序列"""
        self.running = True
        
        try:
            # 1. 发送开始命令
            print("\n=== 测试1: 发送开始命令 ===")
            success = self.send_command("--start")
            if not success:
                print("测试1失败: 无法发送开始命令")
                return False
            
            # 等待可能的识别结果
            print("等待可能的识别结果...")
            time.sleep(5)
            
            # 2. 依次切换到每个组
            for group_id in range(1, 6):
                print(f"\n=== 测试2.{group_id}: 切换到group{group_id} ===")
                success = self.send_command(f"--switch group{group_id}")
                if not success:
                    print(f"测试2.{group_id}失败: 无法切换到group{group_id}")
                    return False
                
                # 等待可能的识别结果
                print("等待可能的识别结果...")
                time.sleep(5)
            
            # 3. 发送停止命令
            print("\n=== 测试3: 发送停止命令 ===")
            success = self.send_command("--stop")
            if not success:
                print("测试3失败: 无法发送停止命令")
                return False
            
            # 等待系统停止处理
            time.sleep(2)
            
            # 4. 发送心跳检测
            print("\n=== 测试4: 发送心跳检测 ===")
            success = self.send_command("--ping")
            if not success:
                print("测试4失败: 无法发送心跳检测")
                return False
            
            print("\n所有测试完成!")
            return True
            
        except Exception as e:
            print(f"测试序列出错: {e}")
            return False
            
        finally:
            self.running = False
    
    def run(self):
        """运行完整测试"""
        try:
            # 启动结果监听器
            if not self.start_result_listener():
                return False
            
            # 启动系统
            if not self.start_system():
                return False
            
            # 创建并启动测试线程
            self.test_thread = threading.Thread(target=self.run_test_sequence)
            self.test_thread.start()
            
            # 设置超时
            self.test_thread.join(timeout=self.timeout)
            
            if self.test_thread.is_alive():
                print(f"测试超时 ({self.timeout}秒)")
                self.running = False
                self.test_thread.join(timeout=5)
                return False
            
            return True
            
        finally:
            # 无论如何，确保停止所有进程
            self.stop_processes()

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='手影识别系统集成测试工具')
    
    parser.add_argument('--timeout', type=int, default=60,
                      help='测试超时时间（秒），默认为60')
    parser.add_argument('--system-cmd', type=str,
                      default=f"{sys.executable} main.py --camera=0 --show",
                      help='启动系统的命令')
    
    return parser.parse_args()

def signal_handler(sig, frame):
    """处理信号"""
    print("\n收到中断信号，退出中...")
    sys.exit(0)

def main():
    """主函数"""
    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    args = parse_args()
    
    # 创建系统测试
    test = SystemTest(args.system_cmd, args.timeout)
    
    # 运行测试
    success = test.run()
    
    # 设置退出代码
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
