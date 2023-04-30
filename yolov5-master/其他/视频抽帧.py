
import cv2
import os
from PIL import Image
video_path = r'F:\新建文件夹/ultralytics_yolov5-master\data\视频/1.mp4'  # 视频地址
output_path = r'F:\新建文件夹/ultralytics_yolov5-master\data\images/'  # 输出文件夹



def Video2Pic():
    videoPath = r"F:\study\yolov5-master-new\yolov5-master\data/1.mp4"  # 读取视频路径
    imgPath = r"F:\study\yolov5-master-new\yolov5-master\data\images"  # 保存图片路径

    cap = cv2.VideoCapture(videoPath)
    # fps = cap.get(cv2.CAP_PROP_FPS)  # 获取帧率
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取宽度
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取高度
    suc = cap.isOpened()  # 是否成功打开
    frame_count = 0
    while suc:
        frame_count += 1
        suc, frame = cap.read()
        cv2.imwrite(imgPath + str(frame_count).zfill(4), frame)
        cv2.waitKey(1)
    cap.release()
    print("视频转图片结束！")
Video2Pic()