import cv2  #OpenCV包
import numpy as np
import os
import struct
# 首先确定原图片的基本信息：数据格式，行数列数，通道数
height=480
weight=640
channel=1
for i in range(1,2):
# 利用numpy的fromfile函数读取raw文件，并指定数据格式
    # data=open('/root/evosklecopy/cam'+str(i)+'.raw',mode="rb")
    # files=open('/root/evosklecopy/cam'+str(i)+'.raw',mode="rb")
    data=open('/root/evosklecopy/cam3(1).raw',mode="rb")
    files=open('/root/evosklecopy/cam3(1).raw',mode="rb")
    files.seek(0, os.SEEK_END)
    filelen=files.tell()
    j=0
    print(j)
    total=0
    while True:
        print("total:{}".format(total))
        if data.tell() >= filelen:
            break
        data.seek(12,1)  #移动当文件第p个字节处，绝对位置
        print(data.tell())
        data1=data.read(4)
        count = len(data1)/4 
        integers = struct.unpack('i'*int(count), data1)
        print("integers:{}".format(integers))
        data.seek(48,1)
        print(data.tell())
        dataImg=data.read(integers[0])
        img_buffer_numpy = np.frombuffer(dataImg, dtype=np.uint8)  # 将 图片字节码bytes  转换成一维的numpy数组 到缓存中
        img_numpy = cv2.imdecode(img_buffer_numpy, 1) 
        imgData = img_buffer_numpy.reshape( height,weight, channel)
        output_dir = "/root/evosklecopy/cam"+str(0)+"/"
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(output_dir+str(j)+".png", imgData)
        j+=1
        print(j)
        print(data.tell())
        total+=integers[0]+64
        

