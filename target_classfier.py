#!/usr/bin/env python
# coding: utf-8

import cv2
import socket
import numpy as np
import torch

import torch.nn as nn
# cv2.namedWindow('original', cv2.WINDOW_NORMAL)
def receive_img(sock, count):
    buf = b''
    while count:
        new_buf = sock.recv(count)
        if not new_buf:
            return None
        buf += new_buf
        count -= len(new_buf)
    return buf

def recv_arrayimg(socket, shape, dtype):
    # 接收数组数据
    count = int(np.prod(shape))
    data = socket.recv(count * dtype(1).itemsize)
    array = np.frombuffer(data, dtype=dtype).reshape(shape)
    return array

def main():
    img_path = "./datasets/ImageNet/tiny-imagenet-200/test/images/test_4.JPEG"

    ip = 'localhost'
    port = 6002

    # 建立socket客户端，连接主机
    sock = socket.socket()
    sock.connect((ip, port))

    # 图像二进制编码
    img = cv2.imread(img_path)

    # #图片转成tensor，并计算预测概率分布
    # tensor_img = transf(img).to(device)
    # tensor_img = torch.unsqueeze(tensor_img,dim=0)

    # # 计算预测概率
    # target_model.eval()
    # pred = soft_max(target_model(tensor_img))
    # print(pred)

    # pred_result = pred.cpu().detach().numpy()
    # cv2.imshow('original', img)
    # encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    # result, img_encode = cv2.imencode('.jpg', img, encode_param)
    result, img_encode = cv2.imencode('.jpg', img)
    data = np.array(img_encode)
    str_data = data.tobytes()

    #发送要脆弱化的图片和初步的预测概率
    sock.send(str(len(str_data)).ljust(16).encode())
    sock.send(str_data)
    # sock.send(pred_result.tobytes())
    i = 0
    while True:

        #接收返回来的图片
        # length = receive_img(sock, 16)
        # str_data = receive_img(sock, int(length))
        adv_img = recv_arrayimg(sock,(64,64,3),np.float32)
        i += 1
        if i>10000:
            break
        #接收二进制数据流解码
        # data = np.frombuffer(str_data, dtype='uint8')
        # decode_img = cv2.imdecode(data, 1)
        print(adv_img)

        #得到预测概率分布并发送回去
        # sock.send(pred_result.tobytes())
    #关闭通信
    sock.close()



if __name__ == "__main__":
    main()
    pass
