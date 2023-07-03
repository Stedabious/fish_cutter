# fish_cutter
本程式使用Yolov7進行辨識魚頭和魚尾，將位置訊息、深度圖和魚隻重量帶入演算法計算出分切線段，將換算後的分切距離傳送至arduino，進行實體分切。

yolov7使用的程式來源為 https://github.com/bubbliiiing/yolov7-pytorch


##   套件版本
```
cudnn == 10.0 (7.6.5.32)
pyqt5 == 5.15.6
torch == 1.7.1+cu110
torchvision == 0.8.2+cu110
numpy == 1.21.2
opencv-python == 4.5.3.56
pandas == 1.3.5
matplotlib == 3.4.3
pillow == 8.3.2
```

##   使用流程
-   確定插入*D435i或L515*深度攝影機和*arduino*再進行開啟
-   以下開始程式步驟

main_test.py 為使用opencv的視窗來進行測試

>  按下s進行拍照
>>  按下s重新拍照，按下d進行yolov7辨識和演算法計算分切距離
>>>  按下d重新拍照，按下f進行arduino切魚

start.py 為適用使用者介面進行測試

![GITHUB]( picture/readme.png "使用者介面")


##  內容

### 功能程式 import_script

-   fish_process_new.py
判斷魚頭朝左邊右邊
魚身體部位截圖

-   area_weight.py
魚片分切演算法

### yolov7檔案
model_data  
nets  
utils

### 使用者介面程式
controller_new.py  
start.py  
UI_login.py  
UI_run.py  

### 測試程式
資料夾中有測試深度攝影機的程式

