import os
import numpy as np
import torch
import cv2

def fish_dir(results):
    k=0

def run(image, fn, depth):
    # image = addGaussianNoise(image, 0.3)
    

    # image =cv2.imread('depth_ROI.png')
    oriDim = image.shape
    image = cv2.resize(image, dsize=(513,513)) - MI
    image = image.astype(np.float32) / 255.
    image = image[:, :, ::-1]
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    for i in range(3):
        image[:, :, i] = image[:, :, i] - means[i]
        image[:, :, i] = image[:, :, i] / stds[i]

    image = torch.from_numpy(image.transpose((2, 0, 1)).astype(np.float32)).float().unsqueeze(0)

    if cuda:
        image = image.cuda()
        
    with torch.no_grad():
        output = model(image)
        output = output.data.cpu().numpy()
        prediction = np.argmax(output, axis=1)[0]

        for cid, c in enumerate(classes):
            mask = np.zeros((prediction.shape[0], prediction.shape[1]), np.uint8) +255
            mask[prediction == cid+1] = 0
            # mask = cv2.morphologyEx(255-mask, cv2.MORPH_OPEN, kernel)
            # mask = 255-cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel2) 

            mask = cv2.resize(mask,dsize=(oriDim[1],oriDim[0]), interpolation=cv2.INTER_NEAREST)
            mask1=mask-255
            max_region =255-select_max_region(mask1)*255

            if c=="head":
                gray_img0 =255-(max_region)
            elif c=="body":
                gray_img1 =255-(max_region)
            elif c=="tail":
                gray_img2 =255-(max_region)
            cv2.imwrite('result/' + c + '/' + fn+'.png', max_region)

        label_pre = np.dstack([gray_img0,gray_img1,gray_img2])
        label_pre = label_pre.astype(np.uint8)

        # segmap = decode_segmap(prediction, dataset='pascal')
        # segmap = (segmap*255).astype(np.uint8)
        # segmap = cv2.resize(segmap,dsize=(oriDim[1],oriDim[0]))
        # segmap = segmap[:, :, ::-1]
        img = label_pre

        # 計算魚尾、身、頭的切線點和面積

        # 總面積
        total_area = oriDim[0] * oriDim[1]
        # 面積
        tail_area = np.count_nonzero(gray_img0)
        body_area = np.count_nonzero(gray_img1)
        head_area = np.count_nonzero(gray_img2)

        # 計算尾最右邊、頭最左邊的位置 (尾左頭右)
        head_L = np.min(np.where(gray_img0 > 0)[1])
        tail_R = np.max(np.where(gray_img2 > 0)[1])
        # 計算尾最左邊、頭最右邊的位置 (尾右頭左)
        head_R = np.max(np.where(gray_img0 > 0)[1])
        tail_L = np.min(np.where(gray_img2 > 0)[1])

        if(tail_L < head_R):
            tail_cut = tail_R
            head_cut = head_L
        elif(tail_L > head_R):
            tail_cut = tail_L
            head_cut = head_R


        print('tail_precents:{:.3f} %'.format(tail_area / total_area * 100))
        print('body_precents:{:.3f} %'.format(body_area / total_area * 100))
        print('head_precents:{:.3f} %'.format(head_area / total_area * 100))

        segmap = label_pre.copy()
        segmap = cv2.line(segmap, (tail_cut,int(segmap.shape[1]/7)), (tail_cut,int(segmap.shape[1]/2)), (255,255,0), 2)
        segmap = cv2.line(segmap, (head_cut,int(segmap.shape[1]/7)), (head_cut,int(segmap.shape[1]/2)), (255,0,255), 2)

        # image = cv2.line(image, (tail_cut,int(image.shape[1]/7)), (tail_cut,int(image.shape[1]/2)), (255,255,0), 2)
        # image = cv2.line(image, (head_cut,int(image.shape[1]/7)), (head_cut,int(image.shape[1]/2)), (255,0,255), 2)

        # cv2.imwrite('result/color_' + fn + '.png', image)
        cv2.imwrite('test.jpg', segmap)

        output_mask = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        cv2.imwrite('result/_full/' + fn +'.jpg', segmap)
        cv2.imwrite('result/ROI_color_' + fn +'.jpg', img)
        cv2.imwrite('result/ROI_gray_' + fn + '.png', output_mask)

        roi = get_ROI(depth, output_mask, head_cut, tail_cut)

        return roi



def get_ROI(img, roi_head, head_cut, tail_cut):
    rows, cols = roi_head.shape[:2]
 
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/

    ret, roi_head = cv2.threshold(roi_head,10,255,cv2.THRESH_BINARY)
 
    # uint8 -> uint16
    roi_head = roi_head.astype(np.uint16)
    # 可以改成自己想要的深度
    roi_head[roi_head != 0] = 65535

    roi_head[:, head_cut:cols] = 0
    roi_head[:, 0:tail_cut] = 0

    mask_head_ = cv2.bitwise_not(roi_head)

    # 如果遮罩相反就開啟下面這行
    # mask_head_ = cv2.bitwise_and(roi_head, roi_head, mask_head_)

    roi_img_ = cv2.bitwise_and(img, img, roi_head)

    # print(type(roi_img_[0,0]))
    # print(type(mask_head_[0,0]))

    add = cv2.add(mask_head_ , roi_img_)

    
    # print(type(add[0, 0]))
    img[:rows, :cols] = add

    img[img != 65535] *= 100
    img[img == 65535] = 0

    # cv2.namedWindow('123', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('123',img)
    return img