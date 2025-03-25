#coding=utf-8
import cv2
import cv2 as cv
import numpy as np
import mvsdk
import time 
import socket
import platform
from get_hand_video_camera_v2_release import skinmask,getcnthull,compare_contours,overlay_foreground_on_left,send_score_to_ip

def main_loop():


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
    
     
    cnt_frams = 0
    cnt_score_ok = 0



#   ========================camera loop ========================

    # 枚举相机
    DevList = mvsdk.CameraEnumerateDevice()
    nDev = len(DevList)
    if nDev < 1:
        print("No camera was found!")
        return

    for i, DevInfo in enumerate(DevList):
        print("{}: {} {}".format(i, DevInfo.GetFriendlyName(), DevInfo.GetPortType()))
    i = 0 if nDev == 1 else int(input("Select camera: "))
    DevInfo = DevList[i]
    print(DevInfo)

    # 打开相机
    hCamera = 0
    try:
        hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
    except mvsdk.CameraException as e:
        print("CameraInit Failed({}): {}".format(e.error_code, e.message) )
        return

    # 获取相机特性描述
    cap = mvsdk.CameraGetCapability(hCamera)

    # 判断是黑白相机还是彩色相机
    monoCamera = (cap.sIspCapacity.bMonoSensor != 0)

    # 黑白相机让ISP直接输出MONO数据，而不是扩展成R=G=B的24位灰度
    if monoCamera:
        mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
    else:
        mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

    # 相机模式切换成连续采集
    mvsdk.CameraSetTriggerMode(hCamera, 0)

    # 手动曝光，曝光时间30ms
    mvsdk.CameraSetAeState(hCamera, 0)
    mvsdk.CameraSetExposureTime(hCamera, 60 * 1000)

    # 让SDK内部取图线程开始工作
    mvsdk.CameraPlay(hCamera)

    # 计算RGB buffer所需的大小，这里直接按照相机的最大分辨率来分配
    FrameBufferSize = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if monoCamera else 3)

    # 分配RGB buffer，用来存放ISP输出的图像
    # 备注：从相机传输到PC端的是RAW数据，在PC端通过软件ISP转为RGB数据（如果是黑白相机就不需要转换格式，但是ISP还有其它处理，所以也需要分配这个buffer）
    pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)

    while (cv.waitKey(1) & 0xFF) != ord('q'):
        # 从相机取一帧图片
        try:
            pRawData, FrameHead = mvsdk.CameraGetImageBuffer(hCamera, 200)
            mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer, FrameHead)
            mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)

            # windows下取到的图像数据是上下颠倒的，以BMP格式存放。转换成opencv则需要上下翻转成正的
            # linux下直接输出正的，不需要上下翻转
            if platform.system() == "Windows":
                mvsdk.CameraFlipFrameBuffer(pFrameBuffer, FrameHead, 1)
            
            # 此时图片已经存储在pFrameBuffer中，对于彩色相机pFrameBuffer=RGB数据，黑白相机pFrameBuffer=8位灰度数据
            # 把pFrameBuffer转换成opencv的图像格式以进行后续算法处理
            frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(pFrameBuffer)
            frame = np.frombuffer(frame_data, dtype=np.uint8)
            frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth, 1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3) )

            frame = cv.resize(frame, (640,480), interpolation = cv.INTER_LINEAR)
            # cv.imshow("Press q to end", frame)
            
            # ==================== detect hand shadow ===============
            
            img = frame
            if cnt_frams % 5 != 0:
                if cnt_frams % 30 == 0:
                    print('每5帧检测一次,每30帧打印一次此信息')
                continue
            try:
                mask_img = skinmask(img)
                contours, hull = getcnthull(mask_img)
                if contours is None:
                    continue
                similarity_score = compare_contours(template_contours, contours)
                cv.drawContours(img, [contours], -1, (255, 255, 0), 2)
                result_frame = overlay_foreground_on_left(img, mask_img, dog_mask_resized)
                cv.putText(result_frame, f"Similarity: {similarity_score:.2f}", (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1.5, (255, 122, 125), 3, cv.LINE_AA)
                if similarity_score > 65:
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
                cv.imshow("img,press q to quit", result_frame)
                # out.write(result_frame)
            except Exception as e:
                print(f"Error: {e}")
                pass
    
            
        except mvsdk.CameraException as e:
            if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                print("CameraGetImageBuffer failed({}): {}".format(e.error_code, e.message) )

    # 关闭相机
    mvsdk.CameraUnInit(hCamera)

    # 释放帧缓存
    mvsdk.CameraAlignFree(pFrameBuffer)

def main():
    try:
        main_loop()
    finally:
        cv.destroyAllWindows()

main()
