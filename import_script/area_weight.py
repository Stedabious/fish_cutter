import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy



def area(depth, output, fish_kg, cut_kg):
    # image = cv2.imread('result/color_test.png')
    depth = depth.astype(np.float64)
    # print(type(depth[0,0]))
    # depth /= 1000
    test_ = depth.nonzero()
    print(test_)
    depth_min = depth[test_].min()
    depth_max = np.max(depth)
    print(depth_min, depth_max)

    background = 49*40  # 理論上是depth_max

    depth[depth > 0] -= background
    depth = abs(depth)
    threshold = 1*40
    depth[depth < threshold] = 0

    depth_underdiff = copy.deepcopy(depth)
    depth_underdiff[depth_underdiff > 0] -= depth_min

    # depth[depth > 0] -= background
    # depth_diff = abs(depth)
    depth_contour = depth - depth_underdiff

    # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.1), cv2.COLORMAP_JET)
    # cv2.imshow('depth', depth_colormap)
    # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_underdiff, alpha=0.1), cv2.COLORMAP_JET)
    # cv2.imshow('depth_underdiff', depth_colormap)
    # # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_diff, alpha=0.1), cv2.COLORMAP_JET)
    # # cv2.imshow('depth_diff', depth_colormap)
    # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_contour, alpha=0.1), cv2.COLORMAP_JET)
    # cv2.imshow('depth_contour', depth_colormap)

    depth_total = sum(sum(depth_contour))

    # fish_kg = 24591 #g
    # cut_kg = 300

    global_weight = fish_kg / depth_total

    test_global = depth_contour * global_weight
    temp = np.cumsum(test_global,axis = 0)
    col_total = temp[-1,:]
    # diff_max = max(col_total)
    row_total = np.cumsum(col_total)

    near_cut_pos = row_total % cut_kg
    cut_pos = np.array(np.where(np.sign(np.diff(near_cut_pos)) == -1)).flatten()
    # plt.plot(near_cut_pos)
    # plt.show()

    
    for i in range(len(cut_pos)):   
        output_ = cv2.line(output, (cut_pos[i],int(output.shape[1]/7)), (cut_pos[i],int(output.shape[1]/2)), (255,255,0), 1)

    return output_, cut_pos