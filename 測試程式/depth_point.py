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
# config.set_option(rs.option.visual_present, 5)

pipeline.start(config)

# colorizer = rs.colorizer()        
# depth_to_disparity = rs.disparity_transform(True)
# disparity_to_depth = rs.disparity_transform(False)

# ----------
# 此段為限制雷射能量並降低最近可測量距離
profile = pipeline.get_active_profile()
# profile.get_device().query_sensors()[0]  # 0 for depth sensor, 1 for camera sensor
sensor = profile.get_device().query_sensors()[0]
sensor.set_option(rs.option.min_distance, 0)
sensor.set_option(rs.option.enable_max_usable_range, 0)
sensor.set_option(rs.option.laser_power, 5)
# ----------


# loop variable 迴圈參數
frames = []
i = 0
catch = False

# 滑鼠位置
point = (200,200)


# 滑鼠
def mouse_point(event, x, y, args, params):
    global point
    point = (x,y)

cv2.namedWindow('image')
cv2.setMouseCallback('image', mouse_point)


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
        cv2.circle(color_image, point, 4, (0, 0, 255))
        distance = depth_frame.get_distance(point[0],point[1])
        depth_distance = depth_image[point[1]][point[0]]
        cv2.putText(color_image,"{:.3f}m".format(distance), (point[0], point[1] ), cv2.FONT_ITALIC, 0.5, (255, 255, 255), 2)
        print(depth_distance)
        print(distance)
        
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        cv2.circle(depth_colormap, point, 4, (0, 0, 255))
        cv2.putText(depth_image,"{:.3f}m".format(depth_distance), (point[0], point[1] ), cv2.FONT_ITALIC, 0.5, (255, 255, 255), 2)
        

        cv2.imshow('depth', depth_colormap)
        cv2.imshow('image', color_image)
        

        if cv2.waitKey(1) & 0xFF == ord('s'):
            catch = True

        # elif catch:
        #     frames.append(depth_frame)
        #     i += 1
        #     if i == 10:
        #         i = 0
        #         for x in range(10):
        #             frame = frames[x]
        #             # frame = decimation.process(frame)
        #             frame = depth_to_disparity.process(frame)
        #             frame = spatial.process(frame)
        #             frame = temporal.process(frame)
        #             frame = disparity_to_depth.process(frame)
        #             frame = hole_filling.process(frame)
        #         frames = []

        #         # 可視化
        #         colorized_depth = np.asanyarray(colorizer.colorize(frame).get_data())
        #         catch = False
        #         cv2.imshow('more depth filter',colorized_depth)
        #         cv2.waitKey(0)


        elif cv2.waitKey(10)&0xff == ord('q'):
            cv2.destroyAllWindows()
            break

finally:
    pipeline.stop()



# https://blog.csdn.net/qq_25105061/article/details/111312298





