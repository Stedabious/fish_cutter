import os
import numpy as np
import torch
import cv2

def fish_dir(results):
    if(results[0][0,1] < results[0][1,1]):
        head = results[0][1,1]
        tail = results[0][0,3]
        fish_top = results[0][1,3]
        fish_bottom = results[0][0,1]

    else:
        head = results[0][1,3]
        tail = results[0][0,1]
        fish_top = results[0][1,1]
        fish_bottom = results[0][0,3]


    return int(head), int(tail), int(fish_top), int(fish_bottom)


def get_ROI(img, roi_head,head, tail):
    rows, cols = img.shape[:2]
 
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/

    ret, roi_head = cv2.threshold(roi_head,10,255,cv2.THRESH_BINARY)
 
    # uint8 -> uint16
    roi_head = roi_head.astype(np.uint16)
    # 可以改成自己想要的深度
    roi_head[roi_head != 0] = 65535

    roi_head[:, head:cols] = 0
    roi_head[:, 0:tail] = 0

    mask_head_ = cv2.bitwise_not(roi_head)

    # 如果遮罩相反就開啟下面這行
    # mask_head_ = cv2.bitwise_and(roi_head, roi_head, mask_head_)

    roi_img_ = cv2.bitwise_and(img, img, roi_head)

    # print(type(roi_img_[0,0]))
    # print(type(mask_head_[0,0]))

    add = cv2.add(mask_head_ , roi_img_)

    
    # print(type(add[0, 0]))
    img[:rows, :cols] = add

    # img[img != 65535] *= 100
    img[img == 65535] = 0

    # cv2.namedWindow('123', cv2.WINDOW_AUTOSIZE)
    # cv2.imshow('123',img)
    return img