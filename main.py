import os
import sys
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from PyQt5 import QtGui
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QFileDialog

import detect
from mainWindow import Ui_MainWindow
from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams, LoadWebcam
from utils.general import check_img_size, non_max_suppression, scale_coords, increment_path, check_imshow
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, time_sync

import warnings

# 忽略警告
warnings.filterwarnings("ignore")


# 主窗口类
class MainWidget(QMainWindow, Ui_MainWindow):
    # 类的构造器
    def __init__(self, parent=None):
        # 调用父类初始化
        super(MainWidget, self).__init__(parent)
        # 设置主窗口标签
        self.setWindowTitle("人体追踪软件")
        # 窗口界面初始化
        self.setupUi(self)

        # 添加工具栏
        self.toolBar.addAction(self.openImage)
        self.toolBar.addAction(self.openVedio)
        self.toolBar.addAction(self.openCamera)
        self.toolBar.addAction(self.closeCamera)

        # 添加事件响应
        self.openImage.triggered.connect(self.open_image)
        self.openVedio.triggered.connect(self.open_video)
        self.openCamera.triggered.connect(self.open_camera)
        self.closeCamera.triggered.connect(self.close_camera)
        self.saveFile.triggered.connect(self.save_file)
        self.aboutApp.triggered.connect(self.about_app)
        self.closeApp.triggered.connect(self.close)

        # 获取格式化的参数
        self.opt = detect.parse_opt()

        # 使用我们自己训练的模型
        self.opt.weights = detect.ROOT / 'best.pt'

        source = str(self.opt.source)
        self.save_img = not self.opt.nosave and not source.endswith('.txt')  # save inference images
        self.save_dir = increment_path(Path(self.opt.project) / self.opt.name,
                                       exist_ok=self.opt.exist_ok)  # increment run

        # 初始化
        self.device = select_device(self.opt.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        cudnn.benchmark = True

        # 加载训练好的模型
        self.model = attempt_load(self.opt.weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        # 筛选person类目标
        self.names = ['person']
        if self.half:
            self.model.half()  # to FP16
        self.imgsz = check_img_size(self.opt.imgsz, s=self.stride)  # check img_size

        # 保存处理结果
        self.img_out = []
        # 摄像头id
        self.camera_id = '0'
        # 摄像头对象
        self.cap = None
        # 视频帧信息
        self.vid_cap_info = []

    # 打开文件或摄像头并加载图像数据
    def open(self, type):
        typeList = ''
        if type == 'image':
            typeList = 'Image Files(*.png *.jpeg *.jpg *.bmp)'
        elif type == 'video':
            typeList = 'Video Files(*.mp4 *.mov *.avi *.mkv *.flv *.wmv)'
        elif type == 'camera':
            pass
        else:
            QMessageBox.warning(None, '错误', '鬼知道你打开了神马')
            time.sleep(1)
            self.close()

        dataset = None
        # 通过不同的输入源来设置不同的数据加载方式
        if type in ('image', 'video'):
            # 打开文件选择窗口
            __fileName, _ = QFileDialog.getOpenFileName(self, '选择文件', '.', typeList)
            # 文件存在
            if __fileName and os.path.exists(__fileName):
                #  Dataloader
                # 一般是直接从文件目录下直接读取图片或者视频数据
                dataset = LoadImages(__fileName, img_size=self.opt.imgsz, stride=self.stride, auto=True)
            else:
                return
        elif type == 'camera':
            self.opt.view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            #  Dataloader
            # 一般是读取摄像头、网络数数据流
            dataset = LoadStreams(self.camera_id, img_size=self.opt.imgsz, stride=self.stride, auto=True)
            self.cap = dataset.cap
        else:
            QMessageBox.warning(None, '错误', '鬼知道你打开了神马')
            time.sleep(1)
            self.close()

        self.deal(dataset, type)

    # 处理数据并显示
    def deal(self, dataset, type):
        # 通过模型进行预测
        if self.device.type != 'cpu':
            self.model(
                torch.zeros(1, 3, *self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        dt, seen = [0.0, 0.0, 0.0], 0

        self.img_out.clear()
        flag = 1
        # 处理预测数据的每一帧图片
        for path, img, im0s, vid_cap in dataset:
            # 获取视频帧信息
            if flag and type == 'video':
                flag = 0
                self.vid_cap_info.append(vid_cap.get(cv2.CAP_PROP_FPS))
                self.vid_cap_info.append(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.vid_cap_info.append(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            t1 = time_sync()
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            # 归一化
            img = img / 255.0  # 0 - 255 to 0.0 - 1.0
            if len(img.shape) == 3:
                img = img[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # 预测
            visualize = increment_path(self.save_dir / Path(path).stem, mkdir=True) if self.opt.visualize else False
            pred = self.model(img, augment=self.opt.augment, visualize=visualize)[0]
            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, self.opt.classes,
                                       self.opt.agnostic_nms, max_det=self.opt.max_det)
            dt[2] += time_sync() - t3

            # 对每张图的所有目标进行标识
            for i, det in enumerate(pred):
                seen += 1
                if type in ('image', 'video'):
                    p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
                else:
                    p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
                s += '%gx%g ' % img.shape[2:]  # print string

                # 绘制目标标识框
                annotator = Annotator(im0, line_width=self.opt.line_thickness, example=str(self.names))
                if len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    n = (det[:, -1] == 0).sum()
                    s += f"{n} {self.names[0]}{'s' * (n > 1)}, "
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)
                        # 只筛选person做标记
                        if c == 0:
                            label = None if self.opt.hide_labels else (
                                self.names[c] if self.opt.hide_conf else f'{self.names[c]} {conf:.2f}')
                            # 绘制框
                            annotator.box_label(xyxy, label, color=colors(c, True))
                # 显示处理耗时
                print(f'{s}Done. ({t3 - t2:.3f}s)')
                self.statusbar.showMessage(f'{s}Done. ({t3 - t2:.3f}s)', 0)

                # 当前帧图片的处理结果
                im0 = annotator.result()

                # 保存每一帧的处理结果，用于保存为文件
                self.img_out.append(im0)

                # 转换颜色空间，通过Qt窗口显示
                result = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
                QtImg = QtGui.QImage(result, result.shape[1], result.shape[0], result.shape[1] * result.shape[2],
                                     QtGui.QImage.Format_RGB888)
                self.content.setPixmap(QtGui.QPixmap.fromImage(QtImg))
                # 等待，避免过快刷新导致窗口卡死，图像无法显示，但是会导致画面卡顿不连续
                cv2.waitKey(1)

    # 打开图片文件
    def open_image(self):
        self.close_camera()
        self.open('image')

    # 打开视频文件
    def open_video(self):
        self.close_camera()
        self.openImage.setDisabled(True)
        self.openVedio.setDisabled(True)
        self.openCamera.setDisabled(True)
        self.saveFile.setDisabled(True)
        self.open('video')
        self.openImage.setEnabled(True)
        self.openVedio.setEnabled(True)
        self.openCamera.setEnabled(True)
        self.saveFile.setEnabled(True)

    # 打开摄像头
    def open_camera(self):
        self.close_camera()
        self.content.setText("正在打开摄像头。。。请稍等")
        self.openCamera.setDisabled(True)
        self.saveFile.setDisabled(True)
        self.closeCamera.setEnabled(True)
        self.open('camera')

    # 关闭摄像头
    def close_camera(self):
        self.closeCamera.setDisabled(True)
        self.saveFile.setEnabled(True)
        if self.cap:
            self.cap.release()
            self.cap = None
        self.openCamera.setEnabled(True)
        self.content.clear()

    # 保存文件
    def save_file(self):
        # 有处理的结果数据才能保存
        l = len(self.img_out)
        typeList = ''
        if l == 1:
            typeList = 'Image Files(*.png *.jpeg *.jpg *.bmp)'
        elif l > 1:
            typeList = 'Video Files(*.mp4  *.avi)'
        else:
            # 消息提示窗口
            QMessageBox.information(self, '提示', '文件保存失败！')
            return

        # 打开文件保存的选择窗口
        __fileName, _ = QFileDialog.getSaveFileName(self, '保存文件', 'temp', typeList)

        # 文件扩展名校验
        filenames = os.path.splitext(__fileName)
        if len(filenames) >= 2:
            if l == 1 and filenames[1] in ('.png', '.jpeg', '.jpg', '.bmp'):
                # 保存图片文件
                cv2.imwrite(__fileName, self.img_out[0])
                print(__fileName, " save ok!")
                # 消息提示窗口
                QMessageBox.information(self, '提示', '文件保存成功！')
                self.statusbar.showMessage(__fileName + " save ok!", 0)
            elif l > 1 and filenames[1] in ('.mp4', '.avi'):
                # 视频文件编码格式
                fourccs = {'.mp4': cv2.VideoWriter_fourcc(*'MP4V'),
                           '.avi': cv2.VideoWriter_fourcc(*'XVID'), }
                # 保存视频文件
                if self.vid_cap_info:
                    video = cv2.VideoWriter(__fileName, fourccs[filenames[1]], int(self.vid_cap_info[0]),
                                            (int(self.vid_cap_info[1]), int(self.vid_cap_info[2])))
                    for im in self.img_out:
                        video.write(im)
                    video.release()
                    print(__fileName, " save ok!")
                    # 消息提示窗口
                    QMessageBox.information(self, '提示', '文件保存成功！')
                    self.statusbar.showMessage(__fileName + " save ok!", 0)
                else:
                    # 消息提示窗口
                    QMessageBox.information(self, '提示', '文件保存失败！')
            else:
                # 消息提示窗口
                QMessageBox.information(self, '提示', '文件保存失败！')
        else:
            # 消息提示窗口
            QMessageBox.information(self, '提示', '文件保存失败！')

    # 关于
    def about_app(self):
        QMessageBox.information(None, '关于', '人体追踪软件1.0\n\nCopyright © 2021–2099\n\n保留一切权利')


if __name__ == "__main__":
    # 固定的，PyQt程序都需要QApplication对象。sys.argv是命令行参数列表，确保程序可以双击运行
    app = QApplication(sys.argv)
    # 创建窗口
    main = MainWidget()
    # 显示窗口
    main.show()
    # 保证程序完整退出
    sys.exit(app.exec())
