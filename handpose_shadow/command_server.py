"""
命令服务器模块
负责监听和处理来自前端的UDP命令消息
"""

import socket
import json
import threading
import time
from .config import UDP_LISTEN_IP, UDP_LISTEN_PORT, COMMAND_TYPES, STATUS_CODES
from .utils.logging_utils import get_logger
from .network_utils import send_ok, send_error

class CommandServer:
    """命令服务器类，用于接收和处理前端命令"""
    
    def __init__(self, listen_ip=None, listen_port=None, callback=None):
        """
        初始化命令服务器
        
        参数:
            listen_ip (str, 可选): 监听的IP地址，默认使用配置值
            listen_port (int, 可选): 监听的端口，默认使用配置值
            callback (callable, 可选): 接收到命令时调用的回调函数
        """
        self.logger = get_logger("command_server")
        
        # 使用传入的值或默认配置
        self.listen_ip = listen_ip or UDP_LISTEN_IP
        self.listen_port = listen_port or UDP_LISTEN_PORT
        self.callback = callback
        
        # 服务器状态
        self.running = False
        self.server_thread = None
        self.socket = None
        
        self.logger.info(f"CommandServer initialized with listen_ip={self.listen_ip}, "
                        f"listen_port={self.listen_port}")
    
    def start(self):
        """
        启动命令服务器
        
        返回:
            bool: 启动是否成功
        """
        if self.running:
            self.logger.warning("Command server is already running")
            return True
            
        try:
            self.running = True
            
            # 创建UDP套接字
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # 绑定到指定地址和端口
            self.socket.bind((self.listen_ip, self.listen_port))
            
            # 设置超时，使线程可以正常退出
            self.socket.settimeout(1.0)
            
            # 创建并启动监听线程
            self.server_thread = threading.Thread(target=self._listen_loop)
            self.server_thread.daemon = True
            self.server_thread.start()
            
            self.logger.info(f"Command server started and listening on {self.listen_ip}:{self.listen_port}")
            
            return True
            
        except Exception as e:
            self.running = False
            self.logger.error(f"Failed to start command server: {e}")
            return False
    
    def stop(self):
        """
        停止命令服务器
        
        返回:
            bool: 停止是否成功
        """
        if not self.running:
            self.logger.warning("Command server is not running")
            return True
            
        try:
            self.running = False
            
            # 等待线程结束
            if self.server_thread:
                self.server_thread.join(timeout=2.0)
                
            # 关闭套接字
            if self.socket:
                self.socket.close()
                self.socket = None
                
            self.logger.info("Command server stopped")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping command server: {e}")
            return False
    
    def _listen_loop(self):
        """命令监听循环"""
        self.logger.debug("Entering command listen loop")
        
        while self.running:
            try:
                # 接收UDP数据包，最大1024字节
                data, addr = self.socket.recvfrom(1024)
                
                if data:
                    self.logger.debug(f"Received {len(data)} bytes from {addr}")
                    self._handle_command(data, addr)
                    
            except socket.timeout:
                # 超时，继续循环
                continue
                
            except Exception as e:
                if self.running:  # 只有在正常运行时才记录错误
                    self.logger.error(f"Error in command listen loop: {e}")
        
        self.logger.debug("Exiting command listen loop")
    
    def _handle_command(self, data, addr):
        """
        处理接收到的命令
        
        参数:
            data (bytes): 接收到的数据
            addr (tuple): 发送方地址，格式为 (ip, port)
        """
        try:
            # 解析JSON数据
            command = json.loads(data.decode('utf-8'))
            
            # 提取命令类型
            cmd_type = command.get('type')
            
            self.logger.info(f"Received command: {cmd_type} from {addr}")
            
            # 验证命令类型
            if cmd_type not in COMMAND_TYPES.values():
                error_msg = f"Invalid command type: {cmd_type}"
                self.logger.warning(error_msg)
                send_error(error_msg, addr[0], addr[1])
                return
            
            # 如果提供了回调函数，则调用回调函数处理命令
            if self.callback:
                try:
                    self.callback(command, addr)
                    
                    # 发送确认响应
                    send_ok("Command received", addr[0], addr[1])
                    
                except Exception as e:
                    error_msg = f"Error processing command: {e}"
                    self.logger.error(error_msg)
                    send_error(error_msg, addr[0], addr[1])
            else:
                self.logger.warning("No callback provided to handle command")
                send_ok("Command received, but no handler is available", addr[0], addr[1])
            
        except json.JSONDecodeError:
            error_msg = "Invalid JSON data"
            self.logger.warning(f"{error_msg}: {data}")
            send_error(error_msg, addr[0], addr[1])
            
        except Exception as e:
            error_msg = f"Error handling command: {e}"
            self.logger.error(error_msg)
            send_error(error_msg, addr[0], addr[1])
    
    def set_callback(self, callback):
        """
        设置命令回调函数
        
        参数:
            callback (callable): 处理命令的回调函数
        """
        self.callback = callback
        self.logger.debug("Command callback set")

class CommandHandler:
    """命令处理器类，用于处理前端命令"""
    
    def __init__(self):
        """初始化命令处理器"""
        self.logger = get_logger("command_handler")
        
        # 注册的处理函数
        self.handlers = {
            COMMAND_TYPES["START"]: [],
            COMMAND_TYPES["STOP"]: [],
            COMMAND_TYPES["SWITCH_SCENE"]: [],
            COMMAND_TYPES["PING"]: []
        }
    
    def register_handler(self, command_type, handler):
        """
        注册命令处理函数
        
        参数:
            command_type (str): 命令类型
            handler (callable): 处理函数
            
        返回:
            bool: 注册是否成功
        """
        if command_type not in self.handlers:
            self.logger.warning(f"Unknown command type: {command_type}")
            return False
        
        self.handlers[command_type].append(handler)
        self.logger.debug(f"Registered handler for command type: {command_type}")
        return True
    
    def handle_command(self, command, addr):
        """
        处理命令
        
        参数:
            command (dict): 命令数据
            addr (tuple): 发送方地址
            
        返回:
            bool: 处理是否成功
        """
        cmd_type = command.get('type')
        
        if cmd_type not in self.handlers:
            self.logger.warning(f"No handlers for command type: {cmd_type}")
            return False
        
        handlers = self.handlers[cmd_type]
        
        if not handlers:
            self.logger.warning(f"No handlers registered for command type: {cmd_type}")
            return False
        
        # 调用所有注册的处理函数
        for handler in handlers:
            try:
                handler(command, addr)
            except Exception as e:
                self.logger.error(f"Error in command handler: {e}")
                
        return True

# 创建默认的命令处理器
default_handler = CommandHandler()

def create_command_server(callback=None, listen_ip=None, listen_port=None):
    """
    创建并启动命令服务器
    
    参数:
        callback (callable, 可选): 命令回调函数
        listen_ip (str, 可选): 监听IP地址
        listen_port (int, 可选): 监听端口
        
    返回:
        CommandServer: 创建的命令服务器实例
    """
    server = CommandServer(listen_ip, listen_port, callback)
    server.start()
    return server