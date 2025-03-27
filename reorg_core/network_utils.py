import socket
import time
from datetime import datetime

def send_score_to_ip(message, ip_address, udp_port=9890, max_retries=20):
    """
    通过UDP协议发送消息到指定IP和端口
    
    参数:
        message: 要发送的消息
        ip_address: 目标IP地址
        udp_port: 目标端口, 默认为9890
        max_retries: 最大重试次数, 默认为20
    """
    # 创建一个UDP套接字
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # 将分数转换为字符串并进行utf-8编码
    message_bytes = str(message).encode('utf-8')
    
    # 指定发送的IP地址和端口
    server_address = (ip_address, udp_port)
  
    # 获取当前时间
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S') 
    
    # 打印带有时间戳的消息 
    print(f'{current_time} - send to {ip_address} port {udp_port}')
    
    for i in range(max_retries):
        try:
            # 发送消息
            udp_socket.sendto(message_bytes, server_address)
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S') 
            print(f"{current_time} 发送的消息: {message_bytes.decode('utf-8')}")
        except Exception as e:
            print(f"发送失败: {e}")
        time.sleep(0.3)
    
    # 关闭套接字
    udp_socket.close()
