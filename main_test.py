import cv2
import numpy as np
import pyrealsense2 as rs
from PIL import Image
import import_script.fish_process_new as FP_

# import import_script.area as area
import import_script.area_weight as area
from yolo import YOLO

import serial

COM_PORT = 'COM7'  # 請自行修改序列埠名稱
BAUD_RATES = 9600
ser = serial.Serial(COM_PORT, BAUD_RATES)


if __name__ == "__main__":

    yolo = YOLO()
    # 模式 image camera
    mode = 'camera'

    pipeline = rs.pipeline()
    config = rs.config()

    #下面的需要修改一下，變成自己的型號
    # config.enable_device('918512073242')
    # 設定輸入大小
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

    intrinsics = pipeline.get_active_profile().get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    print(intrinsics)



    if mode == 'image':
        i = 0


    if mode == 'camera':
        # loop variable 迴圈參數
        frames = []
        i = 0
        catch = False

        try:
            while True:
                
                frame = pipeline.wait_for_frames()
                depth_frame = frame.get_depth_frame()
                color_frame = frame.get_color_frame()

                if not depth_frame or not color_frame:
                    print('no camera')
                    continue
                
                # print('capture success')
                if catch == False:
                    color_image = np.asanyarray(color_frame.get_data())
                    depth_image = np.asanyarray(depth_frame.get_data())
                    
                cv2.imshow('camera', color_image)
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.1), cv2.COLORMAP_JET)
                cv2.imshow('depth_', depth_colormap)

                if cv2.waitKey(1) & 0xFF == ord('s'):
                    catch = True


                elif catch:

                    # 可視化
                    # cv2.imshow('more depth filter',depth_image)
                    if(cv2.waitKey(0) & 0xFF == ord('s')):
                        catch = False

                    elif(cv2.waitKey(0) & 0xFF == ord('d')):

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
                        head, tail, fish_top, fish_bottom = FP_.fish_dir(results)
                        
                        # 處理魚身深度圖    深度圖
                        # depth_img = FP.get_ROI(depth_image, roi_gray, head, tail)
                        depth_img = FP_.get_ROI(depth_image, head, tail)
                        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.1), cv2.COLORMAP_JET)
                        # cv2.imshow('DEPTH_OUT', depth_colormap)

                        # 計算分切線段      彩色圖+深度圖
                        #                        深度圖    彩色圖   魚體重量   分切重量
                        final, cut_pos = area.area(depth_img, frame, 2500, 300)
                        # area.area(depth_img, frame, 5000, 300)
                        cv2.imshow('final', final)
                        print(cut_pos)

                        if(cv2.waitKey(0) & 0xFF == ord('f')):
                            for cut in cut_pos:
                                # cv2.waitKey(40000)
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
                            print('按任意鍵進行退出...')
                            cv2.waitKey(0)
                            break
                                

                        elif(cv2.waitKey(0) & 0xFF == ord('d')):
                            continue
                    # cv2.waitKey(0)


                elif cv2.waitKey(10)&0xff == ord('q'):
                    break

        finally:
            cv2.destroyAllWindows()
            pipeline.stop() 