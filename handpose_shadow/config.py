"""
手影识别系统配置文件
包含所有可配置参数，包括视频设置、识别参数、网络设置和模板定义
"""

import os

# 路径设置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "groups1to5")
TEMPLATES_DIR = os.path.join(TEMPLATES_DIR, "templates")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# 视频设置
VIDEO_SOURCE = 0  # 0表示默认摄像头
FRAME_WIDTH = 640  # 处理帧的宽度
FRAME_HEIGHT = 480  # 处理帧的高度
FRAME_SKIP = 5  # 每隔多少帧处理一次
SHOW_PREVIEW = True  # 是否显示预览窗口

# 识别参数
SIMILARITY_THRESHOLD = 55  # 默认相似度阈值
CONSECUTIVE_FRAMES = 2  # 连续多少帧匹配才认为成功
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
DEFAULT_GROUP = "group4"

# 手影模板组定义
TEMPLATE_GROUPS = {
    "group1": [  # 城市场景
        {"id": "1001", "file": os.path.join("group1", "human.png"), "name": "Ren_Human_人", "threshold": 55}, # 人
        {"id": "1002", "file": os.path.join("group1", "dog.png"), "name": "Gou_Dog_狗", "threshold": 55}, # 狗
        {"id": "1003", "file": os.path.join("group1", "weasel.png"), "name": "HuangYou_weasel_黄鼬", "threshold": 60}, #黄鼬
        {"id": "1004", "file": os.path.join("group1", "hedgehog.png"), "name": "CiWei_hedgehog_刺猬", "threshold": 57}, #刺猬
        {"id": "1005", "file": os.path.join("group1", "blackbird.png"), "name": "WuDong_blackbird_乌鸫", "threshold": 56}, #乌鸫
    ],
    "group2": [  # 冻原场景
        {"id": "1006", "file": os.path.join("group2", "arctic_wolf.png"), "name": "北极狼", "threshold": 55},
        {"id": "1007", "file": os.path.join("group2", "reindeer.png"), "name": "驯鹿", "threshold": 58},
        {"id": "1008", "file": os.path.join("group2", "ptarmigan.png"), "name": "岩雷鸟", "threshold": 60},
        {"id": "1009", "file": os.path.join("group2", "musk_ox.png"), "name": "麝牛", "threshold": 57},
        {"id": "1010", "file": os.path.join("group2", "arctic_hare.png"), "name": "北极兔", "threshold": 56},
    ],
    "group3": [  # 稀树草原场景
        {"id": "1011", "file": os.path.join("group3", "lion.png"), "name": "狮子", "threshold": 55},
        {"id": "1012", "file": os.path.join("group3", "gemsbok.png"), "name": "高角羚", "threshold": 58},
        {"id": "1013", "file": os.path.join("group3", "elephant.png"), "name": "非洲草原象", "threshold": 60},
        {"id": "1014", "file": os.path.join("group3", "buffalo.png"), "name": "非洲野水牛", "threshold": 57},
        {"id": "1015", "file": os.path.join("group3", "giraffe.png"), "name": "南方长颈鹿", "threshold": 56},
    ],
    "group4": [  # 针叶林场景
        {"id": "1016", "file": os.path.join("group4", "tiger.png"), "name": "东北虎", "threshold": 55},
        {"id": "1017", "file": os.path.join("group4", "brown_bear.png"), "name": "棕熊", "threshold": 58},
        {"id": "1018", "file": os.path.join("group4", "marten.png"), "name": "紫貂", "threshold": 60},
        {"id": "1019", "file": os.path.join("group4", "snake.png"), "name": "棕黑锦蛇", "threshold": 57},
        {"id": "1020", "file": os.path.join("group4", "moose.png"), "name": "驼鹿", "threshold": 56},
    ],
    "group5": [  # 雨林场景
        {"id": "1021", "file": os.path.join("group5", "clouded_leopard.png"), "name": "巽他云豹", "threshold": 55},
        {"id": "1022", "file": os.path.join("group5", "pygmy_marmoset.png"), "name": "蜂猴", "threshold": 58},
        {"id": "1023", "file": os.path.join("group5", "snake.png"), "name": "天堂金花蛇", "threshold": 60},
        {"id": "1024", "file": os.path.join("group5", "bat.png"), "name": "短吻果蝠", "threshold": 57},
        {"id": "1025", "file": os.path.join("group5", "locust.png"), "name": "蝗虫", "threshold": 56},
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