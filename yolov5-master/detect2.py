# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     path/                           # directory
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path
#
import torch
import end_number

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
import time
import pymysql


def txt(number):
    print("现在", number)
    path_root = os.path.dirname(__file__) + "\save"
    path_year = path_root + '\\' + time.strftime("%Y", time.localtime())
    folder = os.path.exists(path_year)
    if not folder:
        os.makedirs(path_year)
    path_data = path_year + '\\' + time.strftime("%m-%d", time.localtime())

    with open(f"{path_data}.txt", "a", encoding="utf-8") as fp:
        fp.write("识别号：%s\t识别时间：%s\t操作人：%s\t场站名：%s" % (number[2], number[1], sign.name,  sign.place ))
        fp.write("\n")


def MYSQL(number):
    # 创建数据库连接
    conn = pymysql.connect(host="localhost", port=3306, user="root", passwd="123456", db="end_hole")
    # 获取一个游标对象
    cursor = conn.cursor()
    # sql语句中，用%s做占位符，参数用一个元组
    sql = "insert into data(识别号, 识别时间, 识别时长, 操作人, 场站名) values(%s,%s,%s,%s,%s)"
    param = (number[2], number[1], "", sign.name,  sign.place )
    # 执行数据库插入
    cursor.execute(sql, param)
    # 提交
    conn.commit()
    # 关闭连接
    conn.close()
    cursor.close()


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    global im0
    global capture
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        # print(source)
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        capture = vid_cap
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # TODO Process predictions
        for i, det in enumerate(pred):  # per image

            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            # print(im0.shape)
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                # print(det[:, :4][0][0],9999 )
                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                # TODO 我的代码

                fyd = 0
                if ui.button_open.text() == "开始":
                    return
                if len(det) >= 20:
                    print("开始执行")
                    # 把坐标输入的number中。
                    number__ = end_number.numder(det, gn)
                    print(number__)
                    if number__[0] == 0:
                        ui.label_word3.setText(f"已经识别次数|  {number__[3]}  |次")
                        ui.label_word2.setText(number__[2])
                    elif number__[0] == 1:
                        ui.tabel.setRowCount(ui.tabel.rowCount()+1)
                        txt(number__)
                        MYSQL(number__)
                        word = ui.label_word1.text()
                        word = int(word) + 1
                        ui.label_word1.setText(str(word))
                        # print("今日次数"+word)
                        ui.label_word3.setText("已经识别次数|  100  |次")
                        ui.label_word2.setText(number__[2])
                        print(ui.tabel.rowCount())
                        for row in range(ui.tabel.rowCount()-2, -1, -1):
                            item1 = ui.tabel.item(row, 0).text()
                            item2 = ui.tabel.item(row, 1).text()
                            item3 = ui.tabel.item(row, 2).text()
                            item4 = ui.tabel.item(row, 3).text()
                            item5 = ui.tabel.item(row, 4).text()
                            ui.tabel.setItem(row + 1, 0, QtWidgets.QTableWidgetItem(item1))
                            ui.tabel.setItem(row + 1, 1, QtWidgets.QTableWidgetItem(item2))
                            ui.tabel.setItem(row + 1, 2, QtWidgets.QTableWidgetItem(item3))
                            ui.tabel.setItem(row + 1, 3, QtWidgets.QTableWidgetItem(item4))
                            ui.tabel.setItem(row + 1, 4, QtWidgets.QTableWidgetItem(item5))
                        ui.tabel.setItem(0, 0, QtWidgets.QTableWidgetItem(number__[2]))
                        ui.tabel.setItem(0, 1, QtWidgets.QTableWidgetItem(number__[1]))
                        ui.tabel.setItem(0, 3, QtWidgets.QTableWidgetItem(sign.name))
                        ui.tabel.setItem(0, 4, QtWidgets.QTableWidgetItem(sign.place))
                        if ui.tabel.item(1, 1).text() == "初始化01":
                            time2 = number__[4]
                        else:
                            time1 = number__[4]
                            time = time1 - time2
                            time2 = time1
                            ui.tabel.setItem(0, 2, QtWidgets.QTableWidgetItem(str(int(time)) + "秒"))

                    print("自行完成")
                for *xyxy, conf, cls in reversed(det):
                    fyd = fyd + 1
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format

                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, fyd, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()

            view_img = False
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        # TODO 视频
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        # ui.label_show.setPixmap(QtGui.QPixmap.fromImage(vid_writer))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'weights/hole.pt',
                        help='model path or triton URL')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob/screen/0(webcam)')
    # parser.add_argument('--source', type=str, default=ROOT / '0', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=50, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', default=True, action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', default=True, action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


# if __name__ == "__main__":
#     opt = parse_opt()
#     main(opt)


if __name__ == '__main__':

    from PyQt5 import QtCore, QtGui, QtWidgets
    import sys
    import cv2
    from PyQt5.QtCore import Qt

    capture = None
    im0 = None

    class user_register(QtWidgets.QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.set_ui()
            self.slot_init()

        def set_ui(self):

            self.resize(500, 400)
            self.label_word1 = QtWidgets.QLabel(self)
            self.label_word1.setText("用户名")
            self.label_word1.move(50, 50)
            self.label_word1.setStyleSheet("font-size:22px")

            self.label_word2 = QtWidgets.QLabel(self)
            self.label_word2.setText('密码')
            self.label_word2.move(50, 100)
            self.label_word2.setStyleSheet("font-size:22px")

            self.label_word3 = QtWidgets.QLabel(self)
            self.label_word3.setText('姓名')
            self.label_word3.move(50, 150)
            self.label_word3.setStyleSheet("font-size:22px")

            self.label_word4 = QtWidgets.QLabel(self)
            self.label_word4.setText('场所')
            self.label_word4.move(50, 200)
            self.label_word4.setStyleSheet("font-size:22px")

            self.label_word5 = QtWidgets.QLabel(self)
            self.label_word5.setText('秘钥')
            self.label_word5.move(50, 250)
            self.label_word5.setStyleSheet("font-size:22px")

            self.UserName = QtWidgets.QLineEdit(self)
            self.UserName.setPlaceholderText("请输入账号")
            self.UserName.move(200, 50)
            self.UserName.setStyleSheet("font-size:22px")

            self.PassWord = QtWidgets.QLineEdit(self)
            self.PassWord.setPlaceholderText("请输入密码")
            self.PassWord.move(200, 100)
            self.PassWord.setStyleSheet("font-size:22px")

            self.Name = QtWidgets.QLineEdit(self)
            self.Name.setPlaceholderText("请输入姓名")
            self.Name.move(200, 150)
            self.Name.setStyleSheet("font-size:22px")

            self.Place = QtWidgets.QLineEdit(self)
            self.Place.setPlaceholderText("请输入场所")
            self.Place.move(200, 200)
            self.Place.setStyleSheet("font-size:22px")

            self.Key = QtWidgets.QLineEdit(self)
            self.Key.setPlaceholderText("请输入秘钥")
            self.Key.move(200, 250)
            self.Key.setStyleSheet("font-size:22px")

            self.button_register = QtWidgets.QPushButton(self)
            self.button_register.setText("注册")
            self.button_register.move(250, 300)
            self.button_register.setStyleSheet("font-size:22px")

        def slot_init(self):
            self.button_register.clicked.connect(self.register_in)

        def register_in(self):
            UserList = []
            if "100" == self.Key.text() and self.UserName.text() != "" and self.Name.text()!="" and self.Place.text()!="" and self.PassWord.text()!="":
                with open("save/password.txt", "r", encoding="utf-8") as fp:
                    print(1)
                    for data in fp:
                        if data == "":
                            return
                        UserList.append(data[data.index("用") + 3:data.index("密")].strip())

                if self.UserName.text() in UserList:
                    QtWidgets.QMessageBox.warning(self, "警告用户名重名", "用户名重名，请重新输入", QtWidgets.QMessageBox.Yes)
                else:
                    QtWidgets.QMessageBox.information(self, "注册成功", "注册成功", QtWidgets.QMessageBox.Yes)
                    # txt
                    with open("save/password.txt", "a", encoding="utf-8") as fp:
                        fp.write(f"用户名{self.UserName.text()}密码{self.PassWord.text()}名称{self.Name.text()}场所{self.Place.text()}")
                        fp.write("\n")
                    # mysql
                    # 创建数据库连接
                    conn = pymysql.connect(host="localhost", port=3306, user="root", passwd="123456", db="end_hole")
                    # 获取一个游标对象
                    cursor = conn.cursor()
                    # sql语句中，用%s做占位符，参数用一个元组
                    sql = "insert into user_list(id, 用户名,密码,姓名,场站名) values(%s,%s,%s,%s,%s)"
                    param = (len(UserList)+1, self.UserName.text(), self.PassWord.text(), self.Name.text(), self.Place.text())
                    # 执行数据库插入
                    cursor.execute(sql, param)
                    # 提交
                    conn.commit()
                    # 关闭连接
                    conn.close()
                    cursor.close()


                    sign.show()
                    register.hide()
            else:
                QtWidgets.QMessageBox.warning(self, "警告", "不能为空，或者秘钥错误\n请重新输入", QtWidgets.QMessageBox.Yes)






    class Ui_SignIn(QtWidgets.QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.set_ui()
            self.slot_init()

        def set_ui(self):
            self.resize(500, 300)
            self.label_word1 = QtWidgets.QLabel(self)
            self.label_word1.setText("用户名")
            self.label_word1.move(50,50)
            self.label_word1.setStyleSheet("font-size:33px")

            self.label_word2 = QtWidgets.QLabel(self)
            self.label_word2.setText('密码')
            self.label_word2.move(50, 150)
            self.label_word2.setStyleSheet("font-size:33px")

            self.UserName = QtWidgets.QLineEdit(self)
            self.UserName.setPlaceholderText("请输入账号")
            self.UserName.move(200, 50)
            self.UserName.setStyleSheet("font-size:25px")

            self.PassWord = QtWidgets.QLineEdit(self)
            self.PassWord.setPlaceholderText("请输入密码")
            self.PassWord.move(200, 150)
            self.PassWord.setStyleSheet("font-size:25px")

            self.button_sign = QtWidgets.QPushButton(self)
            self.button_sign.setText("登录")
            self.button_sign.move(100, 250)
            self.button_sign.setStyleSheet("font-style:SimHzi;font-size:33px")

            self.button_register = QtWidgets.QPushButton(self)
            self.button_register.setText("注册")
            self.button_register.move(300, 250)
            self.button_register.setStyleSheet("font-style:SimHzi;font-size:33px")

        def slot_init(self):
            self.button_sign.clicked.connect(self.sign_in)
            self.button_register.clicked.connect(self.register_in)


        def sign_in(self):
            dict_pass = {}
            with open("save/password.txt", "r", encoding="utf-8") as fp:
                for i in fp:
                    dict_pass[i[i.index("用") + 3:i.index("密")].strip()] = i[i.index("密") + 2:].strip()
            User = str(self.UserName.text())
            Pass = str(self.PassWord.text())

            print(User, Pass)
            if User in dict_pass:
                if Pass == dict_pass[User][:dict_pass[User].index("名")].strip():
                    self.name = dict_pass[User][dict_pass[User].index("名")+2:dict_pass[User].index("场")].strip()
                    self.place = dict_pass[User][dict_pass[User].index("场")+2:].strip()
                    ui.show()
                    self.hide()


        def register_in(self):
            self.hide()
            register.show()



    class Ui_MainWindow(QtWidgets.QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)  # 父类的构造函数
            self.label_show_camera = None
            # self.url = 0
            self.url = "http://192.168.5.75:4747/video"
            self.set_ui()  # 初始化程序界面
            self.slot_init()  # 初始化槽函数

        '''程序界面布局'''

        def set_ui(self):
            self.resize(1500, 1000)
            self.setWindowTitle("执行")

            self.button_open = QtWidgets.QPushButton(self)
            self.button_open.move(10, 10)
            self.button_open.setText("开始")

            self.button_close = QtWidgets.QPushButton(self)
            self.button_close.move(200, 10)
            self.button_close.setText("结束")

            self.main_tabel = QtWidgets.QWidget(self)
            self.main_tabel.setGeometry(10, 500, 1500, 500)

            self.tabel = QtWidgets.QTableWidget(self.main_tabel)
            self.scroll = QtWidgets.QScrollArea()
            self.scroll.setGeometry(10, 500, 1500, 500)
            self.scroll.setWidget(self.tabel)
            self.vbox = QtWidgets.QVBoxLayout(self.main_tabel)
            self.vbox.addStretch(1000)
            self.vbox.addWidget(self.scroll)

            # self.setLayout(self.vbox)
            self.tabel.setGeometry(10, 500, 1500, 500)
            self.tabel.setColumnCount(5)
            self.tabel.setRowCount(3)
            self.tabel.setColumnWidth(1, 300)
            self.tabel.setColumnWidth(0, 300)

            self.tabel.setStyleSheet("font-size:22px")
            self.tabel.setHorizontalHeaderLabels(["识别号", "识别时间", "结束时间", "识别人", "场地名"])
            self.tabel.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
            for row in range(self.tabel.rowCount()):
                for column in range(self.tabel.columnCount()):
                    self.tabel.setItem(row, column, QtWidgets.QTableWidgetItem(f"初始化{row}{column}"))



            self.label_word1 = QtWidgets.QLabel(self)
            self.label_word1.move(1200, 400)
            self.label_word1.setStyleSheet("font-size:44px")
            self.label_word1.setText("0000000000000")

            self.label_word2 = QtWidgets.QLabel(self)
            self.label_word2.move(900, 200)
            self.label_word2.setText("xxxxxxxxxxxx")
            self.label_word2.setFrameShape(QtWidgets.QFrame.Box)
            self.label_word2.setStyleSheet("font-size:44px; border-width:20px")

            self.label_word3 = QtWidgets.QLabel(self)
            self.label_word3.setText("已经识别次数|       |次")
            self.label_word3.move(900, 300)
            self.label_word3.setStyleSheet("font-size:44px")



            self.timer_camera = QtCore.QTimer()
            # url = 0
            # self.capture = cv2.VideoCapture(self.url)
            self.timer_camera.start(50)
            self.label_show_camera = QtWidgets.QLabel(self)
            self.label_show_camera.setFrameShape(QtWidgets.QFrame.Box)
            self.label_show_camera.setFrameShadow(QtWidgets.QFrame.Raised)
            self.label_show_camera.setGeometry(QtCore.QRect(90, 90, 500, 400))  # 调整窗口上下 宽高



            # 文字
            self.label1 = QtWidgets.QLabel(self)
            self.label1.setText("当前识别号")
            self.label1.move(900, 100)
            self.label1.setStyleSheet("font-size:44px")

            self.label2 = QtWidgets.QLabel(self)
            self.label2.setText("今日检测次数:")
            self.label2.move(900, 400)
            self.label2.setStyleSheet("font-size:44px")


        def slot_init(self):
            self.button_open.clicked.connect(
                self.button_open_clicked)  # 若该按键被点击，则调用button_open_camera_clicked()
            self.button_close.clicked.connect(
                self.button_close_clicked)  # 若该按键被点击，则调用close()，注意这个close是父类QtWidgets.QWidget自带的，会关闭程序

        def button_open_clicked(self):
            if self.button_open.text() == "开始":
                self.button_open.setText('结束')
                self.timer_camera.timeout.connect(self.openVideo)
                opt = parse_opt()
                main(opt)

            if self.button_open.text() == "结束":
                self.button_open.setText("开始")
        def button_close_clicked(self):
            sys.exit()

        def openVideo(self):
            # ret, frame = self.capture.read()  # 摄像头读取,ret为是否成功打开摄像头,true,false。 frame为视频的每一帧图像
            # frame = cv2.flip(frame, 1)  # 摄像头是和人对立的，将图像左右调换回来正常显示。
            # frame = cv2.resize(frame, (800, 800))
            # print(im0s)
            # print(myIm)
            if im0 is not None:
                # show = im0
                # ret, frame = capture.read()  # 摄像头读取,ret为是否成功打开摄像头,true,false。 frame为视频的每一帧图像
                # frame = cv2.flip(frame, 1)  # 摄像头是和人对立的，将图像左右调换回来正常显示。
                # frame = cv2.resize(frame, (500, 500))
                show = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
                showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0],
                                         QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
                self.label_show_camera.clear()
                self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))
                self.label_show_camera.show()


    app = QtWidgets.QApplication(sys.argv)  # 固定的，表示程序应用
    ui = Ui_MainWindow()  # 实例化Ui_MainWindow
    sign = Ui_SignIn()
    register = user_register()
    # ui.show()  # 调用ui的show()以显示。同样show()是源于父类QtWidgets.QWidget的
    sign.show()

    sys.exit(app.exec_())  # 不加这句，程序界面会一闪而过

