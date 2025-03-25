import cv2

from core_methods.utils.detect_hand import extract_hand



# 主程序
def main():
    # 打开摄像头
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # 调用手部检测函数
        processed_frame, bboxes, landmarks = extract_hand(frame)

        # 打印边界框和关键点信息（如果需要处理返回的数据）
        if bboxes:
            print(f"检测到 {len(bboxes)} 只手")
            for i, bbox in enumerate(bboxes):
                print(f"手 {i + 1} 的边框：{bbox}")
                print(f"手 {i + 1} 的关键点：{landmarks[i]}")

        # 显示处理后的图像
        cv2.imshow('Hand Detection with Bounding Box', processed_frame)

        # 按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 运行主程序
if __name__ == "__main__":
    main()
