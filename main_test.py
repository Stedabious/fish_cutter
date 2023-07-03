import cv2
import numpy as np
import pyrealsense2 as rs
from PIL import Image
import import_script.fish_process_new as FP_

# import import_script.area as area
import import_script.area_weight as area
from yolo import YOLO
import copy

import serial

COM_PORT = 'COM3'  # 請自行修改序列埠名稱
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
                        # cv2.imshow('yolov7', frame)


                        # 處理魚頭魚尾線段
                        head, tail, fish_head, fish_tail, middle = FP_.fish_dir(results)

                        # depth_colormap__ = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.1), cv2.COLORMAP_JET)
                        # cv2.imshow('DEPTH_OUT_original', depth_colormap__)
                        depth_image_ = copy.deepcopy(depth_image)


                        # 處理魚身深度圖    深度圖
                        # depth_img = FP.get_ROI(depth_image, roi_gray, head, tail)
                        depth_img = FP_.get_ROI(depth_image_, head, tail)
                        depth_colormap_ = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.1), cv2.COLORMAP_JET)
                        cv2.imshow('DEPTH_OUT', depth_colormap_)

                        

                        # 計算分切線段      彩色圖+深度圖
                        #                        深度圖    彩色圖   魚體重量   分切重量
                        final, cut_pos = area.area(depth_img, frame, 2500, 600, head, tail)

                        cut_pos = np.insert(cut_pos, len(cut_pos), fish_head)
                        cut_pos = np.insert(cut_pos, 0, fish_tail)

                        print(cut_pos)
                        cut_pos_new = []

                        depth_image = np.asanyarray(depth_frame.get_data())
                        depth_scale = pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()
                        depth_image = depth_image * depth_scale

                        # 深度換算
                        for i in range(len(cut_pos)-1):
                            print(depth_image[middle, cut_pos[i]])
                            print([cut_pos[i+1], middle])

                            depth_point1 = rs.rs2_deproject_pixel_to_point(intrinsics, [middle, cut_pos[i]], depth_image[middle, cut_pos[i]])
                            point1 = np.array([depth_point1[0], depth_point1[1], depth_point1[2]])

                            depth_point2 = rs.rs2_deproject_pixel_to_point(intrinsics, [middle, cut_pos[i+1]], depth_image[middle, cut_pos[i+1]])
                            point2 = np.array([depth_point2[0], depth_point2[1], depth_point2[2]])

                            distance = np.linalg.norm(point2 - point1)
                            
                            print("Distance ", i+1, ": ", distance, "m")
                            cut_pos_new.append( int(distance*100*128/0.4))


                        cv2.imshow('final', final)
                        
                        print(cut_pos_new)

                        if(cv2.waitKey(0) & 0xFF == ord('f')):
                            for cut in cut_pos_new:
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
                                    if(stage == 5):
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