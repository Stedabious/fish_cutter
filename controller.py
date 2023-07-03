# from asyncio import FastChildWatcher
from multiprocessing.connection import wait
from re import M
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QDockWidget, QListWidget
from PyQt5.QtCore import QThread


import time
import numpy as np
import cv2, glob
import threading

from sqlalchemy import false
# import import_script.HX711 as HX711
import import_script.test as test
import import_script.area_weight as area

# from try_depth_ROI_filter import run as run_
import cv2
import sys

from UI_login import Ui_MainWindow as Ui_login
from UI_run import Ui_MainWindow as Ui_run

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
        
        # 小參數
        self.filename = "123"
        self.catch = False
        self.calculate = False
        self.catch_ten = False
        self.is_catched = False
        self.distance = 0
        # self.weight = self.ui.input_fish_weight.text()
        self.catch_frame_ = []
        self.depth_image_ = []
        self.color_image = []

        
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

        # 操作 
        self.ui.button_file.clicked.connect(self.save_file)
        self.ui.button_logout.clicked.connect(self.log_out)
        self.ui.button_camera.clicked.connect(self.run)
        self.ui.button_run_catch.clicked.connect(self.catch_frame)
        self.ui.button_exit.clicked.connect(self.exit)
        self.ui.button_run.clicked.connect(self.process)
        self.ui.button_image_origin.clicked.connect(self.ori_image)
        self.ui.button_image_cut.clicked.connect(self.cut_image)
        self.ui.button_image_fish.clicked.connect(self.roi_image)

    def save_file(self):
        temp = QFileDialog.getExistingDirectory(self,
                  "Open folder",
                  "./")   
        self.ui.save_path.setText(temp)

    def log_out(self):
        self.hide()
        self.login = Login_Window_controller()
        self.login.show()


    def run(self):
        
        import pyrealsense2 as rs
        print('in')
        self.catch = False

        try:

            # Configure depth and color streams
            pipeline = rs.pipeline()
            config = rs.config()

            #下面的需要修改一下，變成自己的型號
            # config.enable_device('918512073242')

            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)


            # Start streaming
            pipeline.start(config)

            # loop variable 迴圈參數

            frames = []
            
            i = 0
            weight = 0

            # pyQT
            self.ui.button_image_cut.setEnabled(False)
            self.ui.button_image_fish.setEnabled(False)
            self.ui.button_image_origin.setEnabled(False)
            self.ui.button_camera.setEnabled(False)
            self.ui.button_run_catch.setEnabled(True)
            self.ui.button_run.setEnabled(False)


            try:
                while True:
                    # print(~(self.catch), not self.catch, self.catch)

                    # print('in1')
                    start = time.time()
                    frames_ = pipeline.wait_for_frames()
                    self.color_frame = frames_.get_color_frame()
                    self.color_image = np.asanyarray(self.color_frame.get_data())
                    self.depth_frame = frames_.get_depth_frame()
                    depth_image = np.asanyarray(self.depth_frame.get_data())
                    self.distance = np.max(depth_image)
                    # self.catch = False
                    end = time.time()
                    total_time = end - start

                    try:
                        fps = 1 / total_time
                    except:
                        fps = 0

                    if not self.catch:
                        cv2.putText(self.color_image,"{:.1f}fps".format(fps), (0, 30), cv2.FONT_ITALIC, 0.5, (255, 255, 255), 2)
                        cv2.putText(self.color_image,"{:.1f}g".format(weight), (0, 60), cv2.FONT_ITALIC, 0.5, (255, 255, 255), 2)
                        height, width, channel = self.color_image.shape
                        bytesPerline = 3 * width
                        self.qimg = QImage(self.color_image, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
                        self.ui.input_img.setPixmap(QPixmap.fromImage(self.qimg)) 

                    if self.catch_ten:
                        self.depth_image_ = np.asanyarray(self.depth_frame.get_data())
                        self.catch_ten = False
                        self.is_catched = True

                    cv2.waitKey(1)
            finally:
                # Stop streaming
                pipeline.stop()
        except ValueError as e:
            print(e)
            reply = QMessageBox.warning(self, "失敗", "請裝設攝影機")

    def catch_frame(self):
        self.ui.button_camera.setEnabled(True)
        self.ui.button_run.setEnabled(True)
        self.ui.button_run_catch.setEnabled(False)
        print('inc')
        
        self.is_catched = False
        self.catch_ten = True

        self.calculate = False
        self.catch = True
        
    def process(self):
        self.ui.button_camera.setEnabled(True)
        self.ui.button_image_cut.setEnabled(True)
        self.ui.button_image_fish.setEnabled(True)
        self.ui.button_image_origin.setEnabled(True)
        self.ui.button_run.setEnabled(False)

        # try:

        fish = float(self.ui.input_fish_weight.text())
        print(fish)
        cut = float(self.ui.input_weight.text())
        print(cut)

        roi_depth, precent = test.run(self.color_image,self.filename, self.depth_image_)
        cutted, cut_num = area.area(roi_depth, self.color_image, fish, cut)
        
        print(precent)
        self.ui.output_tail.display(precent[0])
        self.ui.output_body.display(precent[1])
        self.ui.output_head.display(precent[2])
        self.ui.output_split.display(cut_num)
        self.ui.output_weight.display(fish)
        self.ui.output_length_2.display(self.distance / 10)
        

        cv2.imwrite('result/z_cut/123.jpg', cutted)
        cv2.waitKey(0)
        # except ValueError as e:
        #     print(e)
        #     reply = QMessageBox.warning(self, "失敗", "請重新拍攝一次")
        # except Exception as e:
        #     print(e)

    def exit(self):
        self.close()
        sys.exit()


    def ori_image(self):
        self.path = "./result/color_123.png"
        self.img = cv2.imread(self.path)
        height, width, channel = self.img.shape
        bytesPerline = 3 * width
        self.qimg = QImage(self.img, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.ui.input_img.setPixmap(QPixmap.fromImage(self.qimg)) 

    def cut_image(self):
        self.path = "./result/z_cut/cut_123.png"
        self.img = cv2.imread(self.path)
        height, width, channel = self.img.shape
        bytesPerline = 3 * width
        self.qimg = QImage(self.img, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.ui.input_img.setPixmap(QPixmap.fromImage(self.qimg)) 

    def roi_image(self):
        self.path = "./result/ROI_color_123.jpg"
        self.img = cv2.imread(self.path)
        height, width, channel = self.img.shape
        bytesPerline = 3 * width
        self.qimg = QImage(self.img, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.ui.input_img.setPixmap(QPixmap.fromImage(self.qimg)) 
    

        

    
    # def open_file(self):     
    #     filename, filetype = QFileDialog.getOpenFileName(self,
    #               "Open file",
    #               "./")                 # start path
    #     print(filename)
    #     self.ui.open_file.setText(filename)
    #     self.path = filename
    #     self.show_img()

# if __name__ == '__main__':
    