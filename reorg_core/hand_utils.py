import numpy as np
import cv2 as cv
import time
from datetime import datetime
import socket

# 生成手掌的皮肤掩码
def skinmask(img):
    hsvim = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower = np.array([0, 48, 80], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")
    skinRegionHSV = cv.inRange(hsvim, lower, upper)
    blurred = cv.blur(skinRegionHSV, (2, 2))
    ret, thresh = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY)
    return thresh

# 获取轮廓和凸包
def getcnthull(mask_img):
    contours, hierarchy = cv.findContours(mask_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # 确保 contours 不为空，避免 max() 为空序列错误
    if contours:
        contours = max(contours, key=lambda x: cv.contourArea(x))
        hull = cv.convexHull(contours)
        return contours, hull
    else:
        return None, None 

# 获取凸缺陷
def getdefects(contours):
    hull = cv.convexHull(contours, returnPoints=False)
    defects = cv.convexityDefects(contours, hull)
    return defects

# 把前景图抠出放到原图左侧，并将背景设为白色
def overlay_foreground_on_left(img, mask_img, template_mask):
    # 使用掩码抠出前景
    foreground = cv.bitwise_and(img, img, mask=mask_img)
    
    # 将前景图的黑色背景（像素值为0的地方）替换为白色
    white_background = np.ones_like(foreground) * 255  # 创建全白背景
    foreground_with_white_bg = np.where(foreground == 0, white_background, foreground)  # 替换黑色部分为白色
    
    # 确保 template_mask 和前景图尺寸一致
    template_mask_resized = cv.resize(template_mask, (foreground_with_white_bg.shape[1], foreground_with_white_bg.shape[0]))

    # 直接将 foreground_with_white_bg 中的所有像素变为绿色
    foreground_with_white_bg[:, :] = [0, 255, 0]  # 设置为全绿色

    # 将前景内容覆盖到 template_mask 背景上
    result_on_template = np.where(mask_img[:, :, np.newaxis] == 255, foreground_with_white_bg, template_mask_resized)

    # 创建一个白色的空白画布，大小为原图两倍宽度，以便显示原图和叠加后的图像
    height, width, _ = img.shape
    new_width = width * 2
    canvas = np.ones((height, new_width, 3), dtype=np.uint8) * 255  # 全白背景

    # 将叠加后的 template_mask_resized 放在左侧
    canvas[:, :width] = result_on_template
    
    # 在右侧放置原图
    canvas[:, width:] = img
    
    return canvas

# 计算手指数量
def count_fingers(img, contours):
    # 检测凸缺陷并计数手指
    defects = getdefects(contours)
    cnt = 0
    
    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i][0]
            start = tuple(contours[s][0])
            end = tuple(contours[e][0])
            far = tuple(contours[f][0])
            a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
            if angle <= np.pi / 2:  # angle less than 90 degrees
                cnt += 1
                cv.circle(img, far, 4, [0, 0, 255], -1)
        if cnt > 0:
            cnt = cnt + 1
            
    return cnt

# 比较轮廓
def compare_contours(template_contours, contours):
    # 比较轮廓
    ret = cv.matchShapes(template_contours, contours, cv.CONTOURS_MATCH_I1, 0.0)
    
    # 将匹配的结果（距离）转化为相似度分数（0-100）
    # 距离越接近0，表示越相似，我们假设相似度为 100 - ret * 常数（这里用 100，调节结果）
    similarity = max(0, 100 - ret * 100) + 10

    return similarity

# 准备模板图像
def prepare_template_mask(mask_path, frame_width, frame_height):
    dog_mask_img = cv.imread(mask_path, cv.IMREAD_UNCHANGED)

    # 对图片进行黑白反转
    dog_mask_img = np.invert(dog_mask_img)

    # 去掉 alpha 通道，只保留 RGB
    if dog_mask_img.shape[2] == 4:  # 检查是否有alpha通道
        dog_mask_img = dog_mask_img[:, :, :3]

    # 把图像自适应放大到 1440，4：3 比例
    aspect_ratio = 4/3
    mask_width = 1440
    mask_height = int(mask_width / aspect_ratio)
    dog_mask_img = cv.resize(dog_mask_img, (mask_width, mask_height))

    # 计算需要填充的上下和左右的边界
    top_padding = (frame_height - mask_height) // 2
    bottom_padding = frame_height - mask_height - top_padding
    left_padding = (frame_width - mask_width) // 2
    right_padding = frame_width - mask_width - left_padding

    # 使用 cv.copyMakeBorder 进行四周对称填充
    dog_mask_resized = cv.copyMakeBorder(dog_mask_img, top_padding, bottom_padding, left_padding, right_padding, 
                                        borderType=cv.BORDER_CONSTANT, value=[255, 255, 255])  # 填充为白色

    # 将输入图像转换为灰度图
    gray_dog_mask_resized = cv.cvtColor(dog_mask_resized, cv.COLOR_BGR2GRAY)

    # 将灰度图二值化
    ret, binary_gray_dog_mask_resized = cv.threshold(gray_dog_mask_resized, 127, 255, cv.THRESH_BINARY)
    template_contours, template_hull = getcnthull(binary_gray_dog_mask_resized)
    
    return dog_mask_resized, template_contours
