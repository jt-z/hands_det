"""
手影识别系统配置文件
包含所有可配置参数，包括视频设置、识别参数、网络设置和模板定义
"""

import os

# 路径设置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# 视频设置
VIDEO_SOURCE = 0  # 0表示默认摄像头
FRAME_WIDTH = 640  # 处理帧的宽度
FRAME_HEIGHT = 480  # 处理帧的高度
FRAME_SKIP = 5  # 每隔多少帧处理一次
SHOW_PREVIEW = True  # 是否显示预览窗口

# 识别参数
SIMILARITY_THRESHOLD = 55  # 默认相似度阈值
CONSECUTIVE_FRAMES = 3  # 连续多少帧匹配才认为成功
MIN_CONTOUR_AREA = 5000  # 最小手部轮廓面积（过滤小噪点）

# 皮肤检测参数
SKIN_LOWER_HSV = [0, 48, 80]  # HSV空间皮肤颜色下限
SKIN_UPPER_HSV = [20, 255, 255]  # HSV空间皮肤颜色上限

# 网络设置
UDP_SEND_IP = "127.0.0.1"  # 结果发送目标IP
UDP_SEND_PORT = 9890  # 结果发送目标端口
UDP_LISTEN_IP = "0.0.0.0"  # 命令监听IP (0.0.0.0表示所有接口)
UDP_LISTEN_PORT = 9891  # 命令监听端口
UDP_MAX_RETRIES = 3  # UDP发送最大重试次数
UDP_RETRY_DELAY = 0.3  # UDP重试间隔(秒)

# 日志设置
LOG_LEVEL = "INFO"  # 日志级别: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_TO_FILE = True  # 是否记录日志到文件
LOG_FILENAME = "hand_shadow.log"  # 日志文件名

# 默认活动组
DEFAULT_GROUP = "group1"

# 手影模板组定义
TEMPLATE_GROUPS = {
    "group1": [
        {"id": "dog", "file": os.path.join("group1", "dog.png"), "name": "狗", "threshold": 55},
        {"id": "cat", "file": os.path.join("group1", "cat.png"), "name": "猫", "threshold": 58},
        {"id": "rabbit", "file": os.path.join("group1", "rabbit.png"), "name": "兔", "threshold": 60},
        {"id": "bird", "file": os.path.join("group1", "bird.png"), "name": "鸟", "threshold": 57},
        {"id": "fish", "file": os.path.join("group1", "fish.png"), "name": "鱼", "threshold": 56},
    ],
    "group2": [
        {"id": "elephant", "file": os.path.join("group2", "elephant.png"), "name": "象", "threshold": 55},
        {"id": "deer", "file": os.path.join("group2", "deer.png"), "name": "鹿", "threshold": 58},
        {"id": "snake", "file": os.path.join("group2", "snake.png"), "name": "蛇", "threshold": 60},
        {"id": "eagle", "file": os.path.join("group2", "eagle.png"), "name": "鹰", "threshold": 57},
        {"id": "turtle", "file": os.path.join("group2", "turtle.png"), "name": "龟", "threshold": 56},
    ],
    "group3": [
        {"id": "dragon", "file": os.path.join("group3", "dragon.png"), "name": "龙", "threshold": 55},
        {"id": "tiger", "file": os.path.join("group3", "tiger.png"), "name": "虎", "threshold": 58},
        {"id": "monkey", "file": os.path.join("group3", "monkey.png"), "name": "猴", "threshold": 60},
        {"id": "camel", "file": os.path.join("group3", "camel.png"), "name": "骆驼", "threshold": 57},
        {"id": "fox", "file": os.path.join("group3", "fox.png"), "name": "狐狸", "threshold": 56},
    ],
    "group4": [
        {"id": "flower", "file": os.path.join("group4", "flower.png"), "name": "花", "threshold": 55},
        {"id": "tree", "file": os.path.join("group4", "tree.png"), "name": "树", "threshold": 58},
        {"id": "mountain", "file": os.path.join("group4", "mountain.png"), "name": "山", "threshold": 60},
        {"id": "house", "file": os.path.join("group4", "house.png"), "name": "房子", "threshold": 57},
        {"id": "sun", "file": os.path.join("group4", "sun.png"), "name": "太阳", "threshold": 56},
    ],
    "group5": [
        {"id": "person", "file": os.path.join("group5", "person.png"), "name": "人", "threshold": 55},
        {"id": "face", "file": os.path.join("group5", "face.png"), "name": "脸", "threshold": 58},
        {"id": "hand", "file": os.path.join("group5", "hand.png"), "name": "手", "threshold": 60},
        {"id": "heart", "file": os.path.join("group5", "heart.png"), "name": "心", "threshold": 57},
        {"id": "star", "file": os.path.join("group5", "star.png"), "name": "星星", "threshold": 56},
    ],
}

# 命令定义
COMMAND_TYPES = {
    "START": "start",
    "STOP": "stop",
    "SWITCH_SCENE": "switch_scene",
    "PING": "ping"
}

# 状态码
STATUS_CODES = {
    "OK": "ok",
    "ERROR": "error"
}