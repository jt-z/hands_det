import cv2 as cv
import numpy as np

def show_mask(mask, window_name="Mask"):
    cv.imshow(window_name, mask)
    key = cv.waitKey(0)
    cv.destroyWindow(window_name)
    return key  # 返回按键值，方便条件控制

def visualize_contour(contour, image=None, window_name="Contour"):
    # 如果没有提供图像，创建一个黑色背景
    if image is None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
    else:
        image = image.copy()  # 避免修改原始图像
    
    # 绘制轮廓
    cv.drawContours(image, [contour], -1, (0, 255, 0), 2)
    
    # 显示边界框
    x, y, w, h = cv.boundingRect(contour)
    cv.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 1)
    
    # 显示轮廓的质心
    M = cv.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv.circle(image, (cx, cy), 5, (0, 0, 255), -1)
    
    # 显示面积
    area = cv.contourArea(contour)
    cv.putText(image, f"Area: {area:.1f}", (10, 30), 
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv.imshow(window_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()