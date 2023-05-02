import cv2 as cv
from skimage.io import imread,imsave,imshow
import numpy as np

img_name = 'a68db89e0d28194f55418ab0419ceb46'

def run(img_name):
    img = imread(r"result\\ROI_color_" + img_name + ".jpg")
    # cv.imshow("ima", img4)

    # img = cv.imread(r"D:\Task2_4\result\Cobia (1).png")
    img1 = cv.imread(r"result\head\\" + img_name + ".png")############图片读取
    img2 = cv.imread(r"result\body\\" + img_name + ".png")############图片读取
    img3 = cv.imread(r"result\tail\\" + img_name + ".png")############图片读取


    gray_img =cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray_img1 =255-cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray_img2 =255-cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    gray_img3 =255-cv.cvtColor(img3, cv.COLOR_BGR2GRAY)
    a=0
    a1=0
    a2=0
    a3=0
    max_x=0
    max_x1=0
    q=[]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if gray_img1[i,j]>1:
                if j>max_x:
                    max_x2=j
                q.append(j)
            if gray_img1[i,j]>1:
                a1=a1+1
                # if j>max_x:
                #     max_x=j
            elif gray_img2[i,j]>1:
                a2=a2+1
                if j>max_x1:
                    max_x1=j
            elif gray_img3[i,j]>1:
                a3=a3+1
                if j>max_x:
                    max_x=j
    area_g=0
    c=0
    b=[]
    total=a1+a2+a3
    q=np.array(q)
    for i in range(max_x,max_x1):
        c=0
        for j in range(img.shape[0]):
            if gray_img2[j,i]==255:
                area_g=6000/total+area_g
                if area_g >=300:
                    area_g=0
                    c=i
                    
                    
        if c>0:
            b.append(c)
    for i in range(len(b)-1):
        img = cv.line(img, (b[i],int(img.shape[1]/7)), (b[i],int(img.shape[1]/2)), (255,0,255), 2)

    area_avg=a1/total*100
    area_avg1=a2/total*100
    area_avg2=a3/total*100
    area_avg=round(area_avg,2)
    area_avg1=round(area_avg1,2)
    area_avg2=round(area_avg2,2)

    # print(f'輪廓面積為:{area}')
    b1= f'{area_avg}'
    b2= f'{area_avg1}'
    b3= f'{area_avg2}'
    b= f'{total}'

    cv.putText(img,'total_area:'+b+'pixel', (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv.LINE_AA)
    cv.putText(img,'head_area:'+b1+'%', (10, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv.LINE_AA)
    cv.putText(img,'body_area:'+b2+'%', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv.LINE_AA)
    cv.putText(img,'tail_area:'+b3+'%', (10, 45), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv.LINE_AA)
    img = cv.line(img, (max_x,int(img.shape[1]/7)), (max_x,int(img.shape[1]/2)), (0,0,255), 2)
    img = cv.line(img, (q.min(),int(img.shape[1]/7)), (q.min(),int(img.shape[1]/2)), (0,255,255), 2)


    cv.imshow("image_1", img)
    imsave("./result/z_cut/" + img_name + ".png",img)
    # cv.imwrite("123.png",img)
    # cv.imshow("image_2", image_2)
    cv.waitKey(0)
    cv.destroyAllWindows()