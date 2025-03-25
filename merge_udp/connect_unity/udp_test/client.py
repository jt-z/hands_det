import socket
import time

def send_score_to_ip(score, ip_address, max_retries=50):
    # 创建一个UDP套接字
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # 将分数转换为字符串并进行utf-8编码
    message = str(score).encode('utf-8')
    
    # 指定发送的IP地址和端口
    server_address = (ip_address, 8234)
    
    for i in range(max_retries):
        try:
            # 发送消息
            udp_socket.sendto(message, server_address)
            print(f"发送的消息: {message.decode('utf-8')}")
        except Exception as e:
            print(f"发送失败: {e}")
        time.sleep(2)
    
    # 关闭套接字
    udp_socket.close()

if __name__ == "__main__":
    score = 85.5  # 替换为你要发送的分数
    # ip_address = "127.0.0.1"  # 替换为目标IP
    ip_address = "192.168.220.128"  # 替换为目标IP
    send_score_to_ip(score, ip_address)
