import socket

def start_udp_server(ip_address, port):
    # 创建一个UDP套接字
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # 绑定到指定的IP地址和端口
    server_address = (ip_address, port)
    udp_socket.bind(server_address)
    
    print(f"UDP服务器正在监听 {ip_address}:{port}")
    
    while True:
        # 接收消息，最多接收1024字节
        data, address = udp_socket.recvfrom(1024)
        print(f"接收到来自 {address} 的消息: {data.decode('utf-8')}")
        
        # 处理接收到的数据
        # 你可以在这里添加其他逻辑来处理收到的消息
        if data.decode('utf-8').lower() == "exit":
            print("收到'exit'指令，服务器即将关闭。")
            break
    
    # 关闭套接字
    udp_socket.close()
    print("服务器已关闭。")

if __name__ == "__main__":
    ip_address = "0.0.0.0"  # 监听所有可用网络接口
    port = 9890 # 替换为你要监听的端口
    start_udp_server(ip_address, port)



# 测试OK： 接收到来自 ('127.0.0.1', 56330) 的消息: 85.5