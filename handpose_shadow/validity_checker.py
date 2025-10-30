#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validity_checker.py
独立的软件有效性检查模块
"""

import sys
import platform
import datetime
from typing import Tuple, Optional


class SoftwareValidityChecker:
    """软件有效性检查类"""
    
    def __init__(self, start_date=None, end_date=None):
        # 默认有效期范围，也可以自定义
        self.start_date = start_date or datetime.date(2025, 10, 21)
        self.end_date = end_date or datetime.date(2025, 11, 6)
        
    def check_system_requirements(self) -> Tuple[bool, str]:
        """检查系统基本要求"""
        try:
            # 检查Python版本
            python_version = sys.version_info
            if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 6):
                return False, f"Python版本过低: {sys.version}，需要Python 3.6以上版本"
            
            # 检查操作系统
            system_info = platform.system()
            if system_info not in ['Windows', 'Linux', 'Darwin']:
                return False, f"不支持的操作系统: {system_info}"
            
            # 检查系统架构
            arch = platform.machine()
            supported_archs = ['x86_64', 'AMD64', 'aarch64', 'arm64']
            if arch not in supported_archs:
                return False, f"不支持的系统架构: {arch}"
            
            return True, "系统检查通过"
            
        except Exception as e:
            return False, f"系统检查时发生错误: {str(e)}"
    
    def check_date_validity(self) -> Tuple[bool, str]:
        """检查当前日期是否在有效期内"""
        try:
            current_date = datetime.date.today()
            
            if current_date < self.start_date:
                days_until_start = (self.start_date - current_date).days
                return False, f"软件尚未到启用时间，还有 {days_until_start} 天启用"
            
            if current_date > self.end_date:
                days_expired = (current_date - self.end_date).days
                return False, f"软件已过期 {days_expired} 天"
            
            remaining_days = (self.end_date - current_date).days + 1
            return True, f"软件在有效期内，剩余 {remaining_days} 天"
            
        except Exception as e:
            return False, f"日期检查时发生错误: {str(e)}"
    
    def get_system_info(self) -> dict:
        """获取详细的系统信息"""
        return {
            'platform': platform.platform(),
            'system': platform.system(),
            'release': platform.release(),
            'machine': platform.machine(),
            'python_version': sys.version,
            'current_date': datetime.date.today().isoformat(),
            'current_time': datetime.datetime.now().isoformat()
        }


# ========== 简化的公共接口 ==========

# 全局实例，避免重复创建
_checker_instance = None

def get_checker():
    """获取检查器实例（单例模式）"""
    global _checker_instance
    if _checker_instance is None:
        _checker_instance = SoftwareValidityChecker()
    return _checker_instance


def is_software_valid() -> bool:
    """
    最简单的检查接口，只返回布尔值
    
    Returns:
        bool: 软件是否可以运行
    """
    checker = get_checker()
    
    # 系统检查
    system_ok, _ = checker.check_system_requirements()
    if not system_ok:
        return False
    
    # 日期检查
    date_ok, _ = checker.check_date_validity()
    return date_ok


def check_software_validity(verbose: bool = False) -> Tuple[bool, str]:
    """
    详细的检查接口，返回状态和信息
    
    Args:
        verbose: 是否输出详细信息到控制台
        
    Returns:
        Tuple[bool, str]: (是否通过检查, 详细信息)
    """
    checker = get_checker()
    
    # 系统检查
    system_ok, system_msg = checker.check_system_requirements()
    if not system_ok:
        if verbose:
            print(f"[系统检查] ✗ {system_msg}")
        return False, f"系统检查失败: {system_msg}"
    
    # 日期检查
    date_ok, date_msg = checker.check_date_validity()
    if not date_ok:
        if verbose:
            print(f"[有效期检查] ✗ {date_msg}")
        return False, f"有效期检查失败: {date_msg}"
    
    if verbose:
        print(f"[系统检查] ✓ {system_msg}")
        print(f"[有效期检查] ✓ {date_msg}")
        print("✅ 所有检查通过")
    
    return True, "所有检查通过"


def get_validity_info() -> dict:
    """
    获取完整的有效性信息
    
    Returns:
        dict: 包含系统信息和检查结果的字典
    """
    checker = get_checker()
    system_ok, system_msg = checker.check_system_requirements()
    date_ok, date_msg = checker.check_date_validity()
    
    return {
        'is_valid': system_ok and date_ok,
        'system_check': {'passed': system_ok, 'message': system_msg},
        'date_check': {'passed': date_ok, 'message': date_msg},
        'system_info': checker.get_system_info()
    }


def require_validity(func):
    """
    装饰器：要求函数执行前通过有效性检查
    
    Usage:
        @require_validity
        def your_function():
            # 这个函数只有通过检查才能执行
            pass
    """
    def wrapper(*args, **kwargs):
        if not is_software_valid():
            raise RuntimeError("软件验证失败，功能无法使用")
        return func(*args, **kwargs)
    return wrapper


# ========== 自定义配置接口 ==========

def set_validity_period(start_date: datetime.date, end_date: datetime.date):
    """
    设置自定义的有效期
    
    Args:
        start_date: 开始日期
        end_date: 结束日期
    """
    global _checker_instance
    _checker_instance = SoftwareValidityChecker(start_date, end_date)


if __name__ == "__main__":
    # 测试代码
    print("=" * 50)
    print("软件有效性检查模块测试")
    print("=" * 50)
    
    # 简单检查
    print(f"软件是否有效: {is_software_valid()}")
    
    # 详细检查
    is_valid, message = check_software_validity(verbose=True)
    print(f"检查结果: {message}")
    
    # 获取完整信息
    info = get_validity_info()
    print(f"完整信息: {info['is_valid']}")