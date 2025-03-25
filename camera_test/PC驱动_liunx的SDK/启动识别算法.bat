@echo off
:: 设置 Python 解释器路径
set PYTHON_PATH=C:\Users\FGD\AppData\Local\Programs\Python\Python313\python.exe

:: 设置 Python 脚本路径
set SCRIPT_PATH=C:\Users\FGD\Desktop\stage2024.11.15\cv_grab.py

:: 启动 Python 脚本
start "" "%PYTHON_PATH%" "%SCRIPT_PATH%"

:: 循环监听按键输入
:LOOP
set /p userInput=Press 'q' to quit the script and exit: 
if /i "%userInput%"=="q" (
    echo Shutting down the Python script...
    taskkill /f /im python.exe
    exit
)
goto LOOP
