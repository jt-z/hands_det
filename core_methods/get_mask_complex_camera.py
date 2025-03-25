import cv2 as cv
import numpy as np
import math

# 手掌检测和手指计数
def detect_hand_and_count_fingers(image):
    # 1. 转换为HSV颜色空间，创建皮肤颜色掩码
    hsvim = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    lower = np.array([0, 48, 80], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")
    skinRegionHSV = cv.inRange(hsvim, lower, upper)
    
    # 2. 对图像进行模糊和二值化处理
    blurred = cv.blur(skinRegionHSV, (2, 2))
    ret, thresh = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY)
    
    # 3. 寻找轮廓
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return image, 0  # 未检测到轮廓
    
    # 获取最大轮廓（即手掌）
    contours = max(contours, key=lambda x: cv.contourArea(x))
    
    # 绘制手掌轮廓
    cv.drawContours(image, [contours], -1, (255, 255, 0), 2)
    
    # 4. 凸包检测
    hull = cv.convexHull(contours, returnPoints=False)
    
    # 5. 凸缺陷检测
    defects = cv.convexityDefects(contours, hull)
    
    # 初始化手指计数器
    if defects is None:
        return image, 0  # 没有检测到缺陷
    
    cnt = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(contours[s][0])
        end = tuple(contours[e][0])
        far = tuple(contours[f][0])
        
        # 计算三边距离
        a = np.linalg.norm(np.array(end) - np.array(start))
        b = np.linalg.norm(np.array(far) - np.array(start))
        c = np.linalg.norm(np.array(end) - np.array(far))
        
        # 使用余弦定理计算角度
        angle = math.acos((b**2 + c**2 - a**2) / (2 * b * c))
        
        # 角度小于90度时，判断为手指
        if angle <= math.pi / 2:
            cnt += 1
            cv.circle(image, far, 4, [0, 0, 255], -1)
    
    # 如果计数大于0，实际上要加1，因为凸缺陷数比手指数少1
    if cnt > 0:
        cnt += 1
    
    # 在图像上显示手指数
    cv.putText(image, f"Fingers: {cnt}", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)
    
    return image, cnt

# 示例用法：逐帧处理视频
def main():
    # 打开摄像头或读取视频文件
    cap = cv.VideoCapture(1)  # 0 表示默认摄像头，可以替换为视频文件路径

    if not cap.isOpened():
        print("无法打开摄像头或视频文件")
        return
    
    while True:
        # 逐帧读取视频
        ret, frame = cap.read()
        if not ret:
            print("无法读取视频帧")
            break
        
        # 检测手掌和计数手指
        result_frame, finger_count = detect_hand_and_count_fingers(frame)
        
        # 显示每一帧的结果
        cv.imshow('Hand Detection and Finger Count', result_frame)
        print(f"Detected fingers: {finger_count}")
        
        # 按下 'q' 键退出循环
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放摄像头并关闭所有窗口
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
