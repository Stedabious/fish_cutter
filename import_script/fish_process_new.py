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


def get_ROI(img, head, tail):
    rows, cols = img.shape[:2]
    head = int(head)
    tail = int(tail)
    img[:, head:cols] = 0
    img[:, 0:tail] = 0

    # cv2.namedWindow('123', cv2.WINDOW_AUTOSIZE)
    # img_ = cv2.applyColorMap(cv2.convertScaleAbs(img, alpha=0.1), cv2.COLORMAP_JET)
    # cv2.imshow('123',img_)
    return img