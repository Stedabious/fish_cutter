import cv2
import numpy as np
import pyrealsense2 as rs

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
# sensor.set_option(rs.option.min_distance, 0)
# sensor.set_option(rs.option.enable_max_usable_range, 0)
# sensor.set_option(rs.option.laser_power, 5)
# ----------



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
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        cv2.imshow('camera', color_image)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            catch = True

        elif catch: 
            # 可視化

            catch = False
            cv2.imshow('more depth filter',depth_image)
            cv2.waitKey(0)


        elif cv2.waitKey(10)&0xff == ord('q'):
            break

finally:
    pipeline.stop()



# https://blog.csdn.net/qq_25105061/article/details/111312298





