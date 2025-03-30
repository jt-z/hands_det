"""
命令服务器测试模块
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

from handpose_shadow.command_server import CommandServer, CommandHandler, create_command_server
from handpose_shadow.config import COMMAND_TYPES

class MockUDPClient:
    """模拟UDP客户端，用于发送命令和接收响应"""
    
    def __init__(self, server_ip='127.0.0.1', server_port=12346):
        """初始化模拟客户端"""
        self.server_ip = server_ip
        self.server_port = server_port
        self.socket = None
        self.responses = []
    
    def connect(self):
        """连接到服务器"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.settimeout(2.0)
        
    def close(self):
        """关闭连接"""
        if self.socket:
            self.socket.close()
            self.socket = None
    
    def send_command(self, command_type, **kwargs):
        """
        发送命令
        
        参数:
            command_type (str): 命令类型
            **kwargs: 命令参数
            
        返回:
            dict: 服务器响应，如果超时则返回None
        """
        if not self.socket:
            self.connect()
        
        # 清空之前的响应
        self.responses = []
        
        # 准备命令
        command = {"type": command_type}
        command.update(kwargs)
        
        # 发送命令
        data = json.dumps(command).encode('utf-8')
        self.socket.sendto(data, (self.server_ip, self.server_port))
        
        # 接收响应
        try:
            response_data, addr = self.socket.recvfrom(1024)
            response = json.loads(response_data.decode('utf-8'))
            self.responses.append(response)
            return response
        except socket.timeout:
            return None
        except Exception as e:
            print(f"Error receiving response: {e}")
            return None

class TestCommandServer(unittest.TestCase):
    """测试命令服务器类"""
    
    def setUp(self):
        """测试准备工作"""
        # 命令接收事件
        self.command_received = threading.Event()
        self.received_commands = []
        
        # 创建命令服务器
        self.server = CommandServer(
            listen_ip='127.0.0.1',
            listen_port=12346,
            callback=self._command_callback
        )
        
        # 创建模拟客户端
        self.client = MockUDPClient(server_port=12346)
    
    def tearDown(self):
        """测试清理工作"""
        self.server.stop()
        self.client.close()
    
    def _command_callback(self, command, addr):
        """命令回调函数"""
        self.received_commands.append((command, addr))
        self.command_received.set()
    
    def test_server_start_stop(self):
        """测试服务器启动和停止"""
        # 启动服务器
        result = self.server.start()
        self.assertTrue(result, "服务器应成功启动")
        self.assertTrue(self.server.running, "服务器应处于运行状态")
        
        # 尝试再次启动
        result = self.server.start()
        self.assertTrue(result, "重复启动应返回成功")
        
        # 停止服务器
        result = self.server.stop()
        self.assertTrue(result, "服务器应成功停止")
        self.assertFalse(self.server.running, "服务器应处于停止状态")
        
        # 尝试再次停止
        result = self.server.stop()
        self.assertTrue(result, "重复停止应返回成功")
    
    def test_receive_command(self):
        """测试接收命令"""
        # 启动服务器
        self.server.start()
        
        # 发送测试命令
        self.client.send_command(COMMAND_TYPES["START"], scene_id="group1")
        
        # 等待命令处理
        received = self.command_received.wait(timeout=2.0)
        
        # 验证结果
        self.assertTrue(received, "应接收到命令")
        self.assertEqual(len(self.received_commands), 1, "应处理一条命令")
        
        command, addr = self.received_commands[0]
        self.assertEqual(command["type"], COMMAND_TYPES["START"], "命令类型应匹配")
        self.assertEqual(command["scene_id"], "group1", "场景ID应匹配")
        self.assertEqual(addr[0], "127.0.0.1", "客户端IP应匹配")
    
    def test_server_response(self):
        """测试服务器响应"""
        # 启动服务器
        self.server.start()
        
        # 发送测试命令并接收响应
        response = self.client.send_command(COMMAND_TYPES["PING"])
        
        # 验证响应
        self.assertIsNotNone(response, "应收到响应")
        self.assertEqual(response["status"], "ok", "状态应为ok")
        self.assertIn("message", response, "应包含消息")
        self.assertIn("timestamp", response, "应包含时间戳")
    
    def test_invalid_command(self):
        """测试无效命令"""
        # 启动服务器
        self.server.start()
        
        # 发送无效命令
        response = self.client.send_command("invalid_type")
        
        # 验证响应
        self.assertIsNotNone(response, "应收到错误响应")
        self.assertEqual(response["status"], "error", "状态应为error")
        self.assertIn("Invalid command type", response["message"], "应报告无效命令类型")
    
    def test_malformed_json(self):
        """测试格式错误的JSON"""
        # 启动服务器
        self.server.start()
        
        # 直接发送非JSON数据
        if not self.client.socket:
            self.client.connect()
            
        self.client.socket.sendto(b"not a json", ("127.0.0.1", 12346))
        
        # 接收响应
        try:
            response_data, addr = self.client.socket.recvfrom(1024)
            response = json.loads(response_data.decode('utf-8'))
            
            # 验证响应
            self.assertEqual(response["status"], "error", "状态应为error")
            self.assertIn("Invalid JSON", response["message"], "应报告无效JSON")
            
        except socket.timeout:
            self.fail("未收到响应")
    
    def test_set_callback(self):
        """测试设置回调函数"""
        # 创建新的回调函数和事件
        new_event = threading.Event()
        new_commands = []
        
        def new_callback(command, addr):
            new_commands.append((command, addr))
            new_event.set()
        
        # 设置新回调
        self.server.set_callback(new_callback)
        
        # 启动服务器
        self.server.start()
        
        # 发送命令
        self.client.send_command(COMMAND_TYPES["START"])
        
        # 等待新回调被调用
        received = new_event.wait(timeout=2.0)
        
        # 验证结果
        self.assertTrue(received, "新回调应被调用")
        self.assertEqual(len(new_commands), 1, "新回调应处理一条命令")
        self.assertEqual(len(self.received_commands), 0, "旧回调不应被调用")

class TestCommandHandler(unittest.TestCase):
    """测试命令处理器类"""
    
    def setUp(self):
        """测试准备工作"""
        self.handler = CommandHandler()
        
        # 命令处理事件和计数
        self.events = {
            COMMAND_TYPES["START"]: threading.Event(),
            COMMAND_TYPES["STOP"]: threading.Event(),
            COMMAND_TYPES["SWITCH_SCENE"]: threading.Event()
        }
        self.counts = {
            COMMAND_TYPES["START"]: 0,
            COMMAND_TYPES["STOP"]: 0,
            COMMAND_TYPES["SWITCH_SCENE"]: 0
        }
    
    def _create_handler(self, cmd_type):
        """创建命令处理函数"""
        def handler(command, addr):
            self.counts[cmd_type] += 1
            self.events[cmd_type].set()
        return handler
    
    def test_register_handler(self):
        """测试注册处理函数"""
        # 注册处理函数
        start_handler = self._create_handler(COMMAND_TYPES["START"])
        stop_handler = self._create_handler(COMMAND_TYPES["STOP"])
        
        result1 = self.handler.register_handler(COMMAND_TYPES["START"], start_handler)
        result2 = self.handler.register_handler(COMMAND_TYPES["STOP"], stop_handler)
        
        # 验证结果
        self.assertTrue(result1, "应成功注册START处理函数")
        self.assertTrue(result2, "应成功注册STOP处理函数")
        
        # 注册无效命令类型
        result3 = self.handler.register_handler("invalid_type", lambda c, a: None)
        self.assertFalse(result3, "注册无效命令类型应失败")
    
    def test_handle_command(self):
        """测试处理命令"""
        # 注册处理函数
        for cmd_type in self.events:
            self.handler.register_handler(cmd_type, self._create_handler(cmd_type))
        
        # 处理START命令
        command1 = {"type": COMMAND_TYPES["START"], "scene_id": "group1"}
        result1 = self.handler.handle_command(command1, ("127.0.0.1", 12345))
        
        # 验证结果
        self.assertTrue(result1, "应成功处理命令")
        self.assertTrue(self.events[COMMAND_TYPES["START"]].is_set(), "START处理函数应被调用")
        self.assertEqual(self.counts[COMMAND_TYPES["START"]], 1, "START处理次数应为1")
        
        # 处理未注册的命令类型
        command2 = {"type": COMMAND_TYPES["PING"]}
        result2 = self.handler.handle_command(command2, ("127.0.0.1", 12345))
        
        # 验证结果
        self.assertFalse(result2, "处理未注册的命令类型应返回False")
    
    def test_multiple_handlers(self):
        """测试多个处理函数"""
        # 计数器
        counter1 = 0
        counter2 = 0
        
        # 定义两个处理函数
        def handler1(cmd, addr):
            nonlocal counter1
            counter1 += 1
        
        def handler2(cmd, addr):
            nonlocal counter2
            counter2 += 1
        
        # 注册两个处理函数
        self.handler.register_handler(COMMAND_TYPES["START"], handler1)
        self.handler.register_handler(COMMAND_TYPES["START"], handler2)
        
        # 处理命令
        command = {"type": COMMAND_TYPES["START"]}
        self.handler.handle_command(command, ("127.0.0.1", 12345))
        
        # 验证结果
        self.assertEqual(counter1, 1, "第一个处理函数应被调用一次")
        self.assertEqual(counter2, 1, "第二个处理函数应被调用一次")

if __name__ == '__main__':
    unittest.main()