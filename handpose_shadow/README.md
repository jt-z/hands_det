# 手影识别系统 (Hand Shadow Recognition System)

这是一个基于OpenCV的手影识别系统，可以实时检测手部并将其与预定义的手影模板进行匹配。系统支持通过UDP与前端系统进行双向通信，能够根据前端指令动态切换待识别的手影组。

## 功能特点

- 实时手部检测和轮廓提取
- 支持多组手影模板（共25个手影分为5组）
- 基于轮廓匹配的手影识别
- 双向UDP通信，支持前端控制
- 多线程设计，保证响应及时
- 可配置的识别参数

## 系统结构

- **config.py**: 配置文件，包含所有可配置参数
- **template_manager.py**: 模板管理器，负责加载和处理手影模板
- **hand_detector.py**: 手部检测器，负责从视频中提取手部轮廓
- **contour_matcher.py**: 轮廓匹配器，负责比较手部轮廓与模板
- **network_utils.py**: 网络工具，负责发送识别结果
- **command_server.py**: 命令服务器，负责接收前端指令
- **main.py**: 主程序，协调各模块工作

## 安装与依赖

本系统依赖以下库：
- OpenCV (cv2)
- NumPy
- Socket (Python内置)
- Threading (Python内置)
- JSON (Python内置)

安装依赖：
```bash
pip install opencv-python numpy
```

## 使用方法

1. 准备手影模板，放入相应的组目录中
2. 编辑config.py配置相关参数
3. 运行主程序：

```bash
python main.py
```

## 前端通信协议

### 前端→后端命令格式
```json
{"type": "start", "scene_id": "group1"}  // 开始识别，使用group1模板
{"type": "stop"}                         // 停止识别
{"type": "switch_scene", "scene_id": "group2"}  // 切换到group2模板组
{"type": "ping"}                         // 心跳检测
```

### 后端→前端消息格式
```json
// 识别结果
{
    "id": "hand1",
    "name": "狗",
    "similarity": 85.5,
    "group": "group1",
    "timestamp": "2025-03-27 15:30:45.123"
}

// 命令确认
{"status": "ok", "message": "命令已接收"}

// 错误响应
{"status": "error", "message": "无效的命令格式"}
```

## 开发与测试

项目包含完整的测试套件，位于tests目录。运行测试：

```bash
cd tests
python -m unittest discover
```