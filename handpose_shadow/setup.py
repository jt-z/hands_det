"""
手影识别系统安装脚本
"""

from setuptools import setup, find_packages

setup(
    name="handpose_shadow",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "opencv-python>=4.5.0",
        "numpy>=1.20.0",
    ],
    author="HandPoseShadow Team",
    author_email="example@example.com",
    description="A system for recognizing hand shadow shapes in real-time",
    keywords="computer vision, opencv, hand detection, contour matching",
    python_requires=">=3.7",
)
