import serial

COM_PORT = 'COM5'
BAUD_RATES = 115200

ser = serial.Serial(COM_PORT, BAUD_RATES)

def run():
    while True:
        data_raw = ser.readline()  # 讀取一行
        data = data_raw.decode()   # 用預設的UTF-8解碼
        # print('接收到的原始資料：', data_raw)
        # print('接收到的資料：', data)

        return float(data)