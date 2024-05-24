#!/usr/bin/env python
# coding: utf-8

import cv2
import socket
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from core.models import vgg19_bn

# cv2.namedWindow('serve', cv2.WINDOW_FREERATIO)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
transf = transforms.ToTensor()
soft_max = nn.Softmax(dim=1)

###加载vgg19_bn-TinyImage模型
target_model = vgg19_bn(num_classes=200)
target_model.to(device)
target_model.load_state_dict(torch.load('./tests/experiments/vgg19_bn_TinyImageNet/ckpt_epoch_1.pth'))


def receive_img(sock, count):
    buf = b''
    while count:
        new_buf = sock.recv(count)
        if not new_buf:
            return None
        buf += new_buf
        count -= len(new_buf)
    return buf

def send_tensorimg(sock,img):
    img = torch.squeeze(img,dim=0)
    img = img.cpu().detach().numpy()
    sock.send(img.tobytes())

def recv_array(socket, shape, dtype):
    # 接收数组数据
    count = int(np.prod(shape))
    data = socket.recv(count * dtype(1).itemsize)
    array = np.frombuffer(data, dtype=dtype).reshape(shape)
    return array


def main():
    ip = 'localhost'
    port = 6002

    # 初始化socket，设置为监听状态
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((ip, port))
    s.listen(True)

    # 等待并接收数据
    conn, address = s.accept()
    length = receive_img(conn, 16)
    str_data = receive_img(conn, int(length))

    # 接收二进制数据流解码
    data = np.frombuffer(str_data, dtype='uint8')
    decode_img = cv2.imdecode(data, 1)
    print(decode_img.shape)

    #保存接收到的图像
    adv_img = transf(decode_img).to(device)
    adv_img = torch.unsqueeze(adv_img, dim=0)
    target_model.eval()
    while True:
        #计算方差损失并更新图像
        adv_img.requires_grad = True
        pred = soft_max(target_model(adv_img))
        # predict_array = recv_array(conn,(1,200),np.float32)
        # predict_array = torch.unsqueeze(torch.tensor(predict_array),dim=0) #转成tensor
        cost = -torch.mean(torch.var(pred))#计算损失
        grad = torch.autograd.grad(cost, adv_img,
                                   retain_graph=False, create_graph=False)[0]
        eps = 0.0001
        adv_img = adv_img + eps * grad.sign()
        adv_img = torch.clamp(adv_img, min=0, max=1).detach()
        # print(predict_array)
        print(-cost)

        #发送回图像
        # adv_img = adv_img.cpu().detach().numpy()
        # print(adv_img.shape)
        # result, img_encode = cv2.imencode('.jpg', adv_img)
        # data = np.array(img_encode)
        send_tensorimg(conn,adv_img)
        # str_data = data.tobytes()
        # conn.send(str(len(str_data)).ljust(16).encode())
        # conn.send(str_data)
        # 如果损失收敛则发送停止，否则发送继续


    #通信结束
    s.close()

    # cv2.imshow('serve', decode_img) #显示图像
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    pass