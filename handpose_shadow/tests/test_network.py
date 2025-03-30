"""
网络工具测试模块
"""

import unittest
import os
import sys
import socket
import json
import threading
import time

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from handpose_shadow.network_utils import ResultSender, send_result, send_status, send_ok, send_error

class MockUDPServer:
    """模拟UDP服务器，用于接收和验证发送的消息"""
    
    def __init__(self, port=12345):
        """初始化模拟服务器"""
        self.port = port
        self.socket = None
        self.running = False
        self.server_thread = None
        self.received_data = []
        self.expected_count = 0
        self.received_event = threading.Event()
    
    def start(self, expected_count=1):
        """
        启动服务器
        
        参数:
            expected_count (int): 期望接收的消息数量
        """
        self.expected_count = expected_count
        self.received_data = []
        self.received_event.clear()
        
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(('127.0.0.1', self.port))
        self.socket.settimeout(0.1)
        
        self.running = True
        self.server_thread = threading.Thread(target=self._receive_loop)
        self.server_thread.daemon = True
        self.server_thread.start()
    
    def stop(self):
        """停止服务器"""
        self.running = False
        if self.server_thread:
            self.server_thread.join(timeout=1.0)
        if self.socket:
            self.socket.close()
    
    def _receive_loop(self):
        """接收消息循环"""
        while self.running:
            try:
                data, addr = self.socket.recvfrom(1024)
                if data:
                    # 解析JSON数据
                    message = json.loads(data.decode('utf-8'))
                    self.received_data.append(message)
                    
                    # 如果接收到足够的消息，设置事件
                    if len(self.received_data) >= self.expected_count:
                        self.received_event.set()
            except socket.timeout:
                pass
            except Exception as e:
                print(f"Error in mock server: {e}")
    
    def wait_for_messages(self, timeout=2.0):
        """
        等待接收到预期数量的消息
        
        参数:
            timeout (float): 超时时间，单位为秒
            
        返回:
            bool: 是否接收到预期数量的消息
        """
        return self.received_event.wait(timeout)

class TestNetworkUtils(unittest.TestCase):
    """测试网络工具类"""
    
    def setUp(self):
        """测试准备工作"""
        self.mock_server = MockUDPServer(port=12345)
        self.sender = ResultSender(
            default_ip='127.0.0.1',
            default_port=12345,
            max_retries=2,
            retry_delay=0.1
        )
    
    def tearDown(self):
        """测试清理工作"""
        self.mock_server.stop()
    
    def test_send_result(self):
        """测试发送识别结果"""
        # 启动模拟服务器
        self.mock_server.start()
        
        # 准备测试数据
        result = {
            "id": "test_id",
            "name": "测试",
            "similarity": 85.5,
            "group": "test_group"
        }
        
        # 发送结果
        success = self.sender.send_result(result)
        
        # 等待接收
        received = self.mock_server.wait_for_messages()
        
        # 验证结果
        self.assertTrue(success, "发送应成功")
        self.assertTrue(received, "应接收到消息")
        self.assertEqual(len(self.mock_server.received_data), 1, "应接收到一条消息")
        
        message = self.mock_server.received_data[0]
        self.assertEqual(message["id"], "test_id", "ID应匹配")
        self.assertEqual(message["name"], "测试", "名称应匹配")
        self.assertEqual(message["similarity"], 85.5, "相似度应匹配")
        self.assertEqual(message["group"], "test_group", "组应匹配")
        self.assertIn("timestamp", message, "应包含时间戳")
    
    def test_send_status(self):
        """测试发送状态消息"""
        # 启动模拟服务器
        self.mock_server.start()
        
        # 发送状态消息
        success = self.sender.send_status_message("ok", "测试消息")
        
        # 等待接收
        received = self.mock_server.wait_for_messages()
        
        # 验证结果
        self.assertTrue(success, "发送应成功")
        self.assertTrue(received, "应接收到消息")
        self.assertEqual(len(self.mock_server.received_data), 1, "应接收到一条消息")
        
        message = self.mock_server.received_data[0]
        self.assertEqual(message["status"], "ok", "状态应匹配")
        self.assertEqual(message["message"], "测试消息", "消息内容应匹配")
        self.assertIn("timestamp", message, "应包含时间戳")
    
    def test_send_multiple_messages(self):
        """测试发送多条消息"""
        # 启动模拟服务器，期望接收3条消息
        self.mock_server.start(expected_count=3)
        
        # 发送3条不同的消息
        self.sender.send_result({"id": "id1", "name": "名称1", "similarity": 80, "group": "group1"})
        self.sender.send_status_message("ok", "消息1")
        self.sender.send_result({"id": "id2", "name": "名称2", "similarity": 90, "group": "group2"})
        
        # 等待接收
        received = self.mock_server.wait_for_messages(timeout=3.0)
        
        # 验证结果
        self.assertTrue(received, "应接收到所有消息")
        self.assertEqual(len(self.mock_server.received_data), 3, "应接收到3条消息")
    
    def test_convenience_functions(self):
        """测试便捷函数"""
        # 启动模拟服务器，期望接收3条消息
        self.mock_server.start(expected_count=3)
        
        # 使用便捷函数发送消息
        send_result({"id": "test", "name": "测试", "similarity": 85, "group": "group"},
                   ip='127.0.0.1', port=12345)
        send_ok("成功消息", ip='127.0.0.1', port=12345)
        send_error("错误消息", ip='127.0.0.1', port=12345)
        
        # 等待接收
        received = self.mock_server.wait_for_messages(timeout=3.0)
        
        # 验证结果
        self.assertTrue(received, "应接收到所有消息")
        self.assertEqual(len(self.mock_server.received_data), 3, "应接收到3条消息")
        
        # 验证各条消息
        self.assertEqual(self.mock_server.received_data[0]["id"], "test", "ID应匹配")
        self.assertEqual(self.mock_server.received_data[1]["status"], "ok", "状态应匹配")
        self.assertEqual(self.mock_server.received_data[2]["status"], "error", "状态应匹配")

if __name__ == '__main__':
    unittest.main()