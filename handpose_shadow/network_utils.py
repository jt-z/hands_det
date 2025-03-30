"""
网络工具模块
负责通过UDP协议发送识别结果
"""

import socket
import json
import time
from datetime import datetime
from .config import UDP_SEND_IP, UDP_SEND_PORT, UDP_MAX_RETRIES, UDP_RETRY_DELAY, STATUS_CODES
from .utils.logging_utils import get_logger, LogPerformance

class ResultSender:
    """结果发送器类，用于发送识别结果到前端"""
    
    def __init__(self, default_ip=None, default_port=None, max_retries=None, retry_delay=None):
        """
        初始化结果发送器
        
        参数:
            default_ip (str, 可选): 默认目标IP地址，默认使用配置值
            default_port (int, 可选): 默认目标端口，默认使用配置值
            max_retries (int, 可选): 最大重试次数，默认使用配置值
            retry_delay (float, 可选): 重试间隔，默认使用配置值
        """
        self.logger = get_logger("result_sender")
        
        # 使用传入的值或默认配置
        self.default_ip = default_ip or UDP_SEND_IP
        self.default_port = default_port or UDP_SEND_PORT
        self.max_retries = max_retries or UDP_MAX_RETRIES
        self.retry_delay = retry_delay or UDP_RETRY_DELAY
        
        self.logger.info(f"ResultSender initialized with default_ip={self.default_ip}, "
                        f"default_port={self.default_port}, max_retries={self.max_retries}")
    
    @LogPerformance()
    def send_result(self, result, ip=None, port=None):
        """
        发送识别结果
        
        参数:
            result (dict): 识别结果，至少包含 'id', 'name', 'similarity', 'group' 字段
            ip (str, 可选): 目标IP地址，默认使用初始化时设置的值
            port (int, 可选): 目标端口，默认使用初始化时设置的值
            
        返回:
            bool: 发送是否成功
        """
        if not result:
            self.logger.warning("Cannot send: empty result")
            return False
        
        # 使用传入的值或默认值
        ip = ip or self.default_ip
        port = port or self.default_port
        
        # 准备消息内容
        message_data = {
            "id": result.get("id", "unknown"),
            "name": result.get("name", "Unknown"),
            "similarity": result.get("similarity", 0.0),
            "group": result.get("group", "unknown"),
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        }
        
        self.logger.debug(f"Sending result: {message_data}")
        
        return self._send_udp_message(message_data, ip, port, self.max_retries, self.retry_delay)
    
    def send_status_message(self, status, message, ip=None, port=None):
        """
        发送状态消息
        
        参数:
            status (str): 状态码，应为 'ok' 或 'error'
            message (str): 消息内容
            ip (str, 可选): 目标IP地址，默认使用初始化时设置的值
            port (int, 可选): 目标端口，默认使用初始化时设置的值
            
        返回:
            bool: 发送是否成功
        """
        # 使用传入的值或默认值
        ip = ip or self.default_ip
        port = port or self.default_port
        
        # 准备消息内容
        message_data = {
            "status": status,
            "message": message,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        }
        
        self.logger.debug(f"Sending status message: {message_data}")
        
        return self._send_udp_message(message_data, ip, port, self.max_retries, self.retry_delay)
    
    def _send_udp_message(self, message_data, ip, port, max_retries, retry_delay):
        """
        发送UDP消息
        
        参数:
            message_data (dict): 消息数据
            ip (str): 目标IP地址
            port (int): 目标端口
            max_retries (int): 最大重试次数
            retry_delay (float): 重试间隔
            
        返回:
            bool: 发送是否成功
        """
        # 创建UDP套接字
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        try:
            # 设置超时时间
            udp_socket.settimeout(1.0)
            
            # 将数据转换为JSON字符串并编码
            message_json = json.dumps(message_data)
            message_bytes = message_json.encode('utf-8')
            
            # 指定目标地址
            server_address = (ip, port)
            
            # 尝试发送，最多重试max_retries次
            success = False
            
            for i in range(max_retries):
                try:
                    # 发送消息
                    udp_socket.sendto(message_bytes, server_address)
                    
                    self.logger.debug(f"Message sent to {ip}:{port}")
                    success = True
                    break
                    
                except socket.error as e:
                    self.logger.warning(f"Failed to send message ({i+1}/{max_retries}): {e}")
                    time.sleep(retry_delay)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error sending UDP message: {e}")
            return False
            
        finally:
            # 关闭套接字
            udp_socket.close()

# 创建默认的结果发送器实例
default_sender = ResultSender()

def send_result(result, ip=None, port=None):
    """
    使用默认发送器发送识别结果，便于直接调用
    
    参数:
        result (dict): 识别结果
        ip (str, 可选): 目标IP地址
        port (int, 可选): 目标端口
        
    返回:
        bool: 发送是否成功
    """
    return default_sender.send_result(result, ip, port)

def send_status(status, message, ip=None, port=None):
    """
    使用默认发送器发送状态消息，便于直接调用
    
    参数:
        status (str): 状态码
        message (str): 消息内容
        ip (str, 可选): 目标IP地址
        port (int, 可选): 目标端口
        
    返回:
        bool: 发送是否成功
    """
    return default_sender.send_status_message(status, message, ip, port)

def send_ok(message, ip=None, port=None):
    """
    发送成功状态消息
    
    参数:
        message (str): 消息内容
        ip (str, 可选): 目标IP地址
        port (int, 可选): 目标端口
        
    返回:
        bool: 发送是否成功
    """
    return send_status(STATUS_CODES['OK'], message, ip, port)

def send_error(message, ip=None, port=None):
    """
    发送错误状态消息
    
    参数:
        message (str): 错误消息
        ip (str, 可选): 目标IP地址
        port (int, 可选): 目标端口
        
    返回:
        bool: 发送是否成功
    """
    return send_status(STATUS_CODES['ERROR'], message, ip, port)