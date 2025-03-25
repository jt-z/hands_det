import numpy as np
import cv2 as cv
import time
import socket 
from datetime import datetime

def skinmask(img):
    hsvim = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower = np.array([0, 48, 80], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")
    skinRegionHSV = cv.inRange(hsvim, lower, upper)
    blurred = cv.blur(skinRegionHSV, (2, 2))
    ret, thresh = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY)
    return thresh

def getcnthull(mask_img):
    contours, hierarchy = cv.findContours(mask_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    if contours:
        contours = max(contours, key=lambda x: cv.contourArea(x))
        hull = cv.convexHull(contours)
        return contours, hull
    else:
        return None, None 

def getdefects(contours):
    hull = cv.convexHull(contours, returnPoints=False)
    defects = cv.convexityDefects(contours, hull)
    return defects

def overlay_foreground_on_left(img, mask_img, template_mask):
    foreground = cv.bitwise_and(img, img, mask=mask_img)
    white_background = np.ones_like(foreground) * 255
    foreground_with_white_bg = np.where(foreground == 0, white_background, foreground)
    template_mask_resized = cv.resize(template_mask, (foreground_with_white_bg.shape[1], foreground_with_white_bg.shape[0]))
    foreground_with_white_bg[:, :] = [0, 255, 0]
    result_on_template = np.where(mask_img[:, :, np.newaxis] == 255, foreground_with_white_bg, template_mask_resized)
    height, width, _ = img.shape
    new_width = width * 2
    canvas = np.ones((height, new_width, 3), dtype=np.uint8) * 255
    canvas[:, :width] = result_on_template
    canvas[:, width:] = img
    return canvas

def count_fingers(contours):
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
            angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
            if angle <= np.pi / 2:
                cnt += 1
                cv.circle(img, far, 4, [0, 0, 255], -1)
        if cnt > 0:
            cnt = cnt + 1

def compare_contours(template_contours, contours):
    ret = cv.matchShapes(template_contours, contours, cv.CONTOURS_MATCH_I1, 0.0)
    similarity = max(0, 100 - ret * 100) + 10
    return similarity

def send_score_to_ip(messege, ip_address, udp_port=9890, max_retries=20):
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    message = str(messege).encode('utf-8')
    server_address = (ip_address, udp_port)
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S') 
    print(f'{current_time} - send to {ip_address}  port {udp_port}')
    for i in range(max_retries):
        try:
            udp_socket.sendto(message, server_address)
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S') 
            print(f"{current_time} 发送的消息: {message.decode('utf-8')}")
        except Exception as e:
            print(f"发送失败: {e}")
        time.sleep(0.3)
    udp_socket.close()

def main():
    dog_mask = 'mask_dog526w512h.png'
    dog_mask_img = cv.imread(dog_mask, cv.IMREAD_UNCHANGED)
    dog_mask_img = np.invert(dog_mask_img)
    dog_mask_img = dog_mask_img[:, :, :3]
    dog_mask_img = cv.resize(dog_mask_img, (1440, 1080))
    frame_width = 1920 
    frame_height = 1080
    mask_height, mask_width = dog_mask_img.shape[:2]
    top_padding = (frame_height - mask_height) // 2
    bottom_padding = frame_height - mask_height - top_padding
    left_padding = (frame_width - mask_width) // 2
    right_padding = frame_width - mask_width - left_padding
    dog_mask_resized = cv.copyMakeBorder(dog_mask_img, top_padding, bottom_padding, left_padding, right_padding, 
                                         borderType=cv.BORDER_CONSTANT, value=[255, 255, 255])
    gray_dog_mask_resized = cv.cvtColor(dog_mask_resized, cv.COLOR_BGR2GRAY)
    ret, binary_gray_dog_mask_resized = cv.threshold(gray_dog_mask_resized, 127, 255, cv.THRESH_BINARY)
    template_contours, template_hull = getcnthull(binary_gray_dog_mask_resized)
    video_name = 'dog.mov'
    # source = 0
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
        if cnt_frams % 5 != 0:
            if cnt_frams % 30 == 0:
                print('每5帧检测一次,每30帧打印一次此信息')
            continue
        if not ret:
            break
        try:
            mask_img = skinmask(img)
            contours, hull = getcnthull(mask_img)
            if contours is None:
                continue
            similarity_score = compare_contours(template_contours, contours)
            cv.drawContours(img, [contours], -1, (255, 255, 0), 2)
            result_frame = overlay_foreground_on_left(img, mask_img, dog_mask_resized)
            cv.putText(result_frame, f"Similarity: {similarity_score:.2f}", (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1.5, (255, 122, 125), 3, cv.LINE_AA)
            if similarity_score > 55:
                cnt_score_ok += 1
                if cnt_score_ok > 3:
                    cv.putText(result_frame, "Success", (50, 200), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv.LINE_AA)
                    ip_address = "127.0.0.1"
                    udp_port = 9890
                    print('设置传递到本地的udp server', ip_address)
                    dog_id = '1002'
                    send_score_to_ip(dog_id, ip_address, udp_port)
                    cnt_score_ok = 0
            else:
                cv.putText(result_frame, "Not ok yet.", (50, 200), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv.LINE_AA)
            time.sleep(0.05)
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
