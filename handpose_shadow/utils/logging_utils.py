"""
日志工具模块
提供统一的日志记录功能
"""

import os
import logging
import sys
from datetime import datetime
from ..config import LOGS_DIR, LOG_LEVEL, LOG_TO_FILE, LOG_FILENAME

# 确保日志目录存在
os.makedirs(LOGS_DIR, exist_ok=True)

# 日志级别映射
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}

def setup_logger(name, level=None, log_to_file=None):
    """
    设置日志记录器
    
    参数:
        name (str): 日志记录器名称
        level (str, 可选): 日志级别，默认使用配置
        log_to_file (bool, 可选): 是否记录到文件，默认使用配置
    
    返回:
        logging.Logger: 配置好的日志记录器
    """
    # 使用传入的值或默认配置
    level = level or LOG_LEVEL
    log_to_file = log_to_file if log_to_file is not None else LOG_TO_FILE
    
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVELS.get(level, logging.INFO))
    
    # 清除已有的处理器
    logger.handlers = []
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # 如果需要，创建文件处理器
    if log_to_file:
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = os.path.join(LOGS_DIR, f"{timestamp}_{LOG_FILENAME}")
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

# 创建默认日志记录器
default_logger = setup_logger("hand_shadow")

def get_logger(name=None):
    """
    获取日志记录器
    
    参数:
        name (str, 可选): 日志记录器名称，如果为None则使用默认记录器
    
    返回:
        logging.Logger: 日志记录器
    """
    if name is None:
        return default_logger
    return setup_logger(f"hand_shadow.{name}")

class LogPerformance:
    """
    性能日志装饰器，用于记录函数执行时间
    
    使用方法:
    @LogPerformance(logger)
    def some_function():
        pass
    """
    
    def __init__(self, logger=None):
        """
        初始化装饰器
        
        参数:
            logger (logging.Logger, 可选): 日志记录器
        """
        self.logger = logger or default_logger
    
    def __call__(self, func):
        """
        装饰器调用
        
        参数:
            func (callable): 被装饰的函数
            
        返回:
            callable: 包装后的函数
        """
        def wrapped(*args, **kwargs):
            start_time = datetime.now()
            result = func(*args, **kwargs)
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds() * 1000  # 毫秒
            self.logger.debug(f"Function {func.__name__} took {execution_time:.2f} ms to execute")
            return result
        return wrapped