import numpy as np
import cv2 as cv
import time
import socket 
from datetime import datetime

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

def count_fingers(contours):
    # 检测凸缺陷并计数手指

    defects = getdefects(contours)
    if defects is not None:
        cnt = 0
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
def compare_contours(template_contours, contours):
    # 比较轮廓
    ret = cv.matchShapes(template_contours, contours, cv.CONTOURS_MATCH_I1, 0.0)
    
    # 将匹配的结果（距离）转化为相似度分数（0-100）
    # 距离越接近0，表示越相似，我们假设相似度为 100 - ret * 常数（这里用 100，调节结果）
    
    similarity = max(0, 100 - ret * 100) + 10

    # if similarity <=10:
    #     similarity = 0

    return similarity


def send_score_to_ip(messege, ip_address,udp_port=9890, max_retries=20):
    # 创建一个UDP套接字
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # 将分数转换为字符串并进行utf-8编码
    message = str(messege).encode('utf-8')
    
    # 指定发送的IP地址和端口
    server_address = (ip_address, udp_port)
  
    # 获取当前时间
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S') 
    
    # 打印带有时间戳的消息 
    print(f'{current_time} - send to {ip_address}  port {udp_port}')
    
    for i in range(max_retries):
        try:
            # 发送消息
            udp_socket.sendto(message, server_address)
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S') 
            print(f"{current_time} 发送的消息: {message.decode('utf-8')}")
        except Exception as e:
            print(f"发送失败: {e}")
        time.sleep(0.3)
    
    # 关闭套接字
    udp_socket.close()


# 主函数
def main():
    dog_mask = 'mask_dog526w512h.png'

    dog_mask_img = cv.imread(dog_mask, cv.IMREAD_UNCHANGED)

    # 对图片进行黑白反转
    dog_mask_img = np.invert(dog_mask_img)

    # 去掉 alpha 通道，只保留 RGB
    dog_mask_img = dog_mask_img[:, :, :3]

    # 把图像自适应放大到 1440，4：3 比例
    dog_mask_img = cv.resize(dog_mask_img, (1440, 1080))

    # 设置摄像头分辨率
    frame_width = 1920 
    frame_height = 1080

    # 计算需要填充的上下和左右的边界
    mask_height, mask_width = dog_mask_img.shape[:2]
    top_padding = (frame_height - mask_height) // 2
    bottom_padding = frame_height - mask_height - top_padding
    left_padding = (frame_width - mask_width) // 2
    right_padding = frame_width - mask_width - left_padding

    # 使用 cv.copyMakeBorder 进行四周对称填充
    dog_mask_resized = cv.copyMakeBorder(dog_mask_img, top_padding, bottom_padding, left_padding, right_padding, 
                                         borderType=cv.BORDER_CONSTANT, value=[255, 255, 255])  # 填充为白色

    # 显示填充后的图像
    # cv.imshow("Resized Dog Mask", dog_mask_resized)



    # 将输入图像转换为灰度图
    gray_dog_mask_resized = cv.cvtColor(dog_mask_resized, cv.COLOR_BGR2GRAY)

    # 将灰度图二值化
    ret, binary_gray_dog_mask_resized = cv.threshold(gray_dog_mask_resized, 127, 255, cv.THRESH_BINARY)
    template_contours, template_hull = getcnthull(binary_gray_dog_mask_resized)


    # 打开摄像头或视频文件
    video_name = 'dog.mov'
    source = video_name  # 视频文件路径
    source = 0  # 视频文件路径

    cap = cv.VideoCapture(source)

    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv.CAP_PROP_FPS))

    out = cv.VideoWriter(f'output_green_{video_name}.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (frame_width * 2, frame_height))

    cnt_frams = 0
    cnt_score_ok = 0
    while cap.isOpened():
        ret, img = cap.read()
        cnt_frams += 1
        if cnt_frams % 5 !=0:
            if cnt_frams % 30 == 0:
                print('每5帧检测一次,每30帧打印一次此信息')
            continue

        if not ret:
            break

        try:
            # 生成皮肤掩码并提取轮廓
            mask_img = skinmask(img)
            contours, hull = getcnthull(mask_img)
            if contours is None:
                continue

            similarity_score = compare_contours(template_contours,contours)

            # 绘制手掌轮廓和凸包
            cv.drawContours(img, [contours], -1, (255, 255, 0), 2)
            # cv.drawContours(img, [hull], -1, (0, 255, 255), 2)
 
            # cnt = count_fingers(contours)
            # cv.putText(img, str(cnt), (0, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)

            # 将前景图放到原图左侧，背景设为白色, 前景手为黑色
            result_frame = overlay_foreground_on_left(img, mask_img, dog_mask_resized)

            # 分数大于60，显示大文字，提示比对成功，并给出分数，否则提示不相似；持续显示similarity_score 
            # 在屏幕上显示相似度分数，位置调整为左上角，下移一些
            cv.putText(result_frame, f"Similarity: {similarity_score:.2f}", (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1.5, (255, 122, 125), 3, cv.LINE_AA)

            # 如果相似度大于 60，显示提示信息 "success"
            if similarity_score > 55:
                cnt_score_ok += 1
                if cnt_score_ok >3:
                    cv.putText(result_frame, "Success", (50, 200), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv.LINE_AA)
                    # 传递信息给udp server
                    ip_address = "127.0.0.1"  # 替换为目标IP
                    udp_port = 9890
                    print('设置传递到本地的udp server', ip_address)
                    dog_id = '1002'
                    send_score_to_ip(dog_id,ip_address,udp_port)
                    cnt_score_ok = 0

            else:
                cv.putText(result_frame, "Not ok yet.", (50, 200), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv.LINE_AA)

            time.sleep(0.05)

            # 显示结果并保存
            cv.imshow("img", result_frame)
            out.write(result_frame)

        except Exception as e:
            print(f"Error: {e}")
            pass

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
