# from asyncio import FastChildWatcher
from multiprocessing.connection import wait
from re import M
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QDockWidget, QListWidget
from PyQt5.QtCore import QThread

from yolo import YOLO
import time
import numpy as np
import cv2, glob
from PIL import Image
import copy

from sqlalchemy import false
# import import_script.HX711 as HX711
import import_script.fish_process_new as FP_

# import import_script.area as area
import import_script.area_weight as area

# from try_depth_ROI_filter import run as run_

import sys
import serial

from UI_login import Ui_MainWindow as Ui_login
from UI_run import Ui_MainWindow as Ui_run

COM_PORT = 'COM7'  # 請自行修改序列埠名稱
BAUD_RATES = 9600
ser = serial.Serial(COM_PORT, BAUD_RATES)


def yolov7(color_image, depth_image, fish_weight, cut_weight):
    yolo = YOLO()
    # ==========yolov7================
    # 格式转变，BGRtoRGB
    frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    frame_ = frame.astype(np.uint8)
    roi_gray = cv2.cvtColor(frame_, cv2.COLOR_RGB2GRAY)
    # 转变成Image
    frame = Image.fromarray(np.uint8(frame))
    # 进行检测
    frame, results = np.array(yolo.detect_image(frame))
    # print(results)
    frame = np.array(frame)
    # RGBtoBGR满足opencv显示格式
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # print(type(frame))
    cv2.imshow('yolov7', frame)

    # 處理魚頭魚尾線段
    head, tail = FP.fish_dir(results)
    
    # 處理魚身深度圖    深度圖
    depth_img = FP_.get_ROI(depth_image, head, tail)

    # 計算分切線段      彩色圖+深度圖
    #           深度圖    彩色圖   魚體重量   分切重量
    final, cut_pos = area.area(depth_img, frame, 2500, 300)
    # area.area(depth_img, frame, 5000, 300)

    cv2.imshow('final', final)
    print(cut_pos)

    return cut_pos, results


class Login_Window_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__() # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_login()
        self.ui.setupUi(self)
        self.setup_control()

    def setup_control(self):
        # 標題圖片
        self.path = "./picture/title.png"
        self.img = cv2.imread(self.path)
        self.img = cv2.resize(self.img, (680, 153))
        height, width, channel = self.img.shape
        bytesPerline = 3 * width
        self.qimg = QImage(self.img, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.ui.title_pic.setPixmap(QPixmap.fromImage(self.qimg))   

        # 按鈕
        self.ui.button_login.clicked.connect(self.login)
        self.ui.button_exit.clicked.connect(exit)



    def login(self):
        # get user & password
        account = self.ui.login_user.text()
        password = self.ui.login_password.text()

        if account == "" or password == "":
            reply = QMessageBox.warning(self, "警告", "請輸入帳號或密碼")
            return

        # 代改
        elif account == "user" and password == "123":
            self.hide()
            self.run = Run_Window_Controller()
            self.run.show()
            
        
        else:
            reply = QMessageBox.warning(self, "警告", "帳號或密碼有誤")


    def exit(self):
        self.close()


class Run_Window_Controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_run()
        self.ui.setupUi(self)
        self.setup_control()

    # 參數宣告
        # 影像
        self.color_frame = []
        self.depth_frame = []
        self.color_image = []
        self.depth_image = []
        self.depth_cal = []

        self.color_image_cut = []

        # 數字
        self.distance = 0
        self.weight = 0
        self.cut_pos = []

        # 開關
        self.iscatched = False

        # 深度攝影機
        self.intrinsics = 0
        self.depth_scale = 0

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        super().closeEvent(a0)
        sys.exit()

    def setup_control(self):
        # 標題圖片
        self.path = "./picture/title.png"
        self.img = cv2.imread(self.path)
        self.img = cv2.resize(self.img, (481, 91))
        height, width, channel = self.img.shape
        bytesPerline = 3 * width
        self.qimg = QImage(self.img, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.ui._title.setPixmap(QPixmap.fromImage(self.qimg))

        # 操作按鈕
        self.ui.button_camera.clicked.connect(self.run)
        self.ui.button_run_catch.clicked.connect(self.catch)
        self.ui.button_run.clicked.connect(self.calculate)
        self.ui.button_run_cut.clicked.connect(self.cut_process)

        self.ui.button_image_origin.clicked.connect(self.ori_image)
        self.ui.button_image_cut.clicked.connect(self.cut_image)

        self.ui.button_logout.clicked.connect(self.log_out)
        self.ui.button_exit.clicked.connect(self.exit)


    # 攝影機 ===================
    def run(self):
        # 按鈕初始化
        self.ui.button_camera.setEnabled(False)
        self.ui.button_run_catch.setEnabled(True)
        self.ui.button_run.setEnabled(False)
        self.ui.button_run_cut.setEnabled(False)

        self.ui.button_image_cut.setEnabled(False)
        self.ui.button_image_origin.setEnabled(False)

        self.iscatched = False

        try:

            import pyrealsense2 as rs
            print('in')

            pipeline = rs.pipeline()
            config = rs.config()

            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

            pipeline.start(config)

            # ----------
            # 此段為限制雷射能量並降低最近可測量距離
            profile = pipeline.get_active_profile()
            # profile.get_device().query_sensors()[0]  # 0 for depth sensor, 1 for camera sensor
            sensor = profile.get_device().query_sensors()[0]
            sensor.set_option(rs.option.min_distance, 0)
            sensor.set_option(rs.option.enable_max_usable_range, 0)
            sensor.set_option(rs.option.laser_power, 50)
            # ----------
            self.intrinsics = pipeline.get_active_profile().get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            self.depth_scale = pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()

            while not self.iscatched:
                
                frame = pipeline.wait_for_frames()
                depth_frame = frame.get_depth_frame()
                color_frame = frame.get_color_frame()
                self.color_image = np.asanyarray(color_frame.get_data())
                self.depth_image = np.asanyarray(depth_frame.get_data())
                height, width, channel = self.color_image.shape
                bytesPerline = 3 * width
                self.qimg = QImage(self.color_image, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
                self.ui.input_img.setPixmap(QPixmap.fromImage(self.qimg)) 


                

                cv2.waitKey(1)



        except ValueError as e:
            print(e)
            reply = QMessageBox.warning(self, "失敗", "請裝設攝影機")

    # 拍照 ===================
    def catch(self):
        # 按鈕初始化
        self.iscatched = True
        self.ui.button_camera.setEnabled(True)
        self.ui.button_run.setEnabled(True)
        self.ui.button_run_catch.setEnabled(False)


    # 計算 ===================
    def calculate(self):
        # 按鈕初始化
        self.ui.button_camera.setEnabled(True)
        self.ui.button_run.setEnabled(False)
        self.ui.button_run_cut.setEnabled(True)
        import pyrealsense2 as rs

        yolo = YOLO()

        frame = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)
        frame_ = frame.astype(np.uint8)
        roi_gray = cv2.cvtColor(frame_, cv2.COLOR_RGB2GRAY)
        # 转变成Image
        frame = Image.fromarray(np.uint8(frame))
        # 进行检测
        frame, results = np.array(yolo.detect_image(frame))
        # print(results)
        frame = np.array(frame)
        # RGBtoBGR满足opencv显示格式
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # print(type(frame))
        # cv2.imshow('yolov7', frame)
        print(len(results))

        cut_weight = self.ui.combo_weight.currentText()
        weight = self.ui.input_fish_weight.text()

        if(len(results[0]) != 2):
            QMessageBox.warning(self, "辨識失敗", "請重新拍攝")
            self.ui.button_run_cut.setEnabled(False)

        else:

            head, tail, fish_head, fish_tail, middle = FP_.fish_dir(results)
            depth_image_ = copy.deepcopy(self.depth_image)

            self.depth_cal = FP_.get_ROI(depth_image_, head, tail)
            final, cut_pos = area.area(self.depth_cal, frame, int(weight), int(cut_weight), head, tail)
            self.color_image_cut = final

            cut_pos = np.insert(cut_pos, len(cut_pos), fish_head)
            cut_pos = np.insert(cut_pos, 0, fish_tail)

            cut_pos_new = []

            # depth_image = np.asanyarray(self.depth_frame.get_data())
            
            depth_image = self.depth_image * self.depth_scale

            for i in range(len(cut_pos)-1):
                # print(depth_image_[middle, cut_pos[i]])
                # print([cut_pos[i+1], middle])

                depth_point1 = rs.rs2_deproject_pixel_to_point(self.intrinsics, [middle, cut_pos[i]], depth_image[middle, cut_pos[i]])
                point1 = np.array([depth_point1[0], depth_point1[1], depth_point1[2]])

                depth_point2 = rs.rs2_deproject_pixel_to_point(self.intrinsics, [middle, cut_pos[i+1]], depth_image[middle, cut_pos[i+1]])
                point2 = np.array([depth_point2[0], depth_point2[1], depth_point2[2]])

                distance = np.linalg.norm(point2 - point1)
                
                print("Distance ", i+1, ": ", distance, "m")
                cut_pos_new.append( int(distance*100*128/0.4))

            self.cut_pos = cut_pos_new


            # 顯示原始、分切圖像
            self.ui.button_image_cut.setEnabled(True)
            self.ui.button_image_origin.setEnabled(True) 

            self.ui.output_split.display(len(cut_pos))
            self.ui.output_camera_distance.display(49)
            self.ui.output_weight.display(weight)

    # 進行分切 ===================
    def cut_process(self):
        # 按鈕初始化
        self.ui.button_run_cut.setEnabled(False)

        for cut in self.cut_pos:
            print('input: ' + str(cut))
            cut_ = str(cut).encode()
            ser.write(cut_)
            stage=0

            while True:
                mcu_feedback = ser.readline().decode('utf-8')  # 接收回應訊息並解碼
                print('控制板回應：', mcu_feedback)
                stage += 1
                # print(stage)
                if(stage == 6):
                    break

        self.ui.button_camera.setEnabled(True)

    # def catch_button(self):
    #     self.ui.button_camera.setEnabled(True)
    #     self.ui.button_run.setEnabled(True)
    #     self.ui.button_run_catch.setEnabled(False)

    # def calculate_button(self):

        
    # def cut_process_button(self):






    def log_out(self):
        self.hide()
        self.login = Login_Window_controller()
        self.login.show()


    def exit(self):
        self.close()
        sys.exit()


    def ori_image(self):
        # self.path = "./result/color_123.png"
        # self.img = cv2.imread(self.path)
        self.img = self.color_image
        height, width, channel = self.img.shape
        bytesPerline = 3 * width
        self.qimg = QImage(self.img, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.ui.input_img.setPixmap(QPixmap.fromImage(self.qimg)) 

    def cut_image(self):
        # self.path = "./result/z_cut/cut_123.png"
        # self.img = cv2.imread(self.path)
        self.img = self.color_image_cut
        height, width, channel = self.img.shape
        bytesPerline = 3 * width
        self.qimg = QImage(self.img, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.ui.input_img.setPixmap(QPixmap.fromImage(self.qimg)) 

