import os.path
import pandas as pd
import math
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import random
import core
import visdom

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def predresult(model,img):
    model.eval()
    pred_score = model(img)
    return pred_score.argmax(1)

def mycompare(a, b):
    index = 0
    size = a.shape
    for x, y in zip(a, b):
        if x==y:
            index += 1
    return index/size[0]

def train_model(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_model(dataloader, model):
    size = len(dataloader.dataset)
    # num_batches = len(dataloader)
    model.eval()
    correct = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%\n")

# target_model = core.models.ResNet(18).to(device)
# target_model.load_state_dict(torch.load('./tests/experiments/ResNet-18_CIFAR-10/ckpt_epoch_100.pth'))
target_model = core.models.ResNet(18)
target_model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
target_model.to(device)
target_model.load_state_dict(torch.load('./tests/experiments/Resnet_Mnist/ckpt_epoch_100.pth'))
batch_size = 8
setup_seed(66)

tr_transformer = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
    # transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
])
te_transformer = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
])
# train_data = datasets.CIFAR10(root='./datasets', train=True, download=True, transform=te_transformer)
# test_data = datasets.CIFAR10(root='./datasets', train=False, download=True, transform=te_transformer)
train_data = datasets.MNIST(root='./datasets', train=True, download=True, transform=te_transformer)
test_data = datasets.MNIST(root='./datasets', train=False, download=True, transform=te_transformer)
train_dataloader = DataLoader(train_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)
test_model(test_dataloader,target_model)#测试模型精度
size = len(train_dataloader.dataset)

# for batch, (X, y) in enumerate(train_dataloader):
#     X, y = X.to(device), y.to(device)
#     Test_img = X[1].clone()
#     Test_label = y[1].clone()
#     break
# vis = visdom.Visdom(env='sen_img')
# loss = nn.CrossEntropyLoss()
# soft_max = nn.Softmax(dim=1)
# Test_img.requires_grad = True
# vis.images(Test_img)
# target_model.eval()
# for i in range(100):
#     Test_img.requires_grad = True
#     outputs = target_model(torch.unsqueeze(Test_img,dim=0))
#     print(outputs)
#     print(Test_label)
#     print(soft_max(outputs))
#     # print(torch.var(soft_max(outputs)))
#     # exit()
#     cost = -loss(outputs, torch.unsqueeze(Test_label,dim=0)) - 100*torch.mean(torch.var(soft_max(outputs)))
#     print(torch.mean(torch.var(soft_max(outputs))))
#     print(soft_max(outputs))
#     print(Test_label)
#     # Update adversarial images
#     grad = torch.autograd.grad(cost, Test_img,
#                                retain_graph=False, create_graph=False)[0]
#     eps = 1/255
#     Test_img = Test_img + eps * grad.sign()
#     Test_img = torch.clamp(Test_img, min=0, max=1).detach()
#     now_outputs = target_model(torch.unsqueeze(Test_img,dim=0))
# print(torch.mean(torch.var(soft_max(now_outputs))))
# print(soft_max(now_outputs))
# vis.images(Test_img)

for batch, (X, y) in enumerate(train_dataloader):
    X, y = X.to(device), y.to(device)
    Test_img = X.clone()
    Test_label = y.clone()
    break
vis = visdom.Visdom(env='sen_img')
loss = nn.CrossEntropyLoss()
soft_max = nn.Softmax(dim=1)
Test_img.requires_grad = True
vis.images(Test_img)
target_model.eval()
for i in range(100):
    Test_img.requires_grad = True
    outputs = target_model(Test_img)
    print(outputs)
    print(Test_label)
    print(soft_max(outputs))
    # print(torch.var(soft_max(outputs)))
    # exit()
    cost = -0*loss(outputs, Test_label) - 100*torch.mean(torch.var(soft_max(outputs)))
    print(torch.mean(torch.var(soft_max(outputs))))
    print(soft_max(outputs))
    print(Test_label)
    # Update adversarial images
    grad = torch.autograd.grad(cost, Test_img,
                               retain_graph=False, create_graph=False)[0]
    eps = 1/255
    Test_img = Test_img + eps * grad.sign()
    Test_img = torch.clamp(Test_img, min=0, max=1).detach()
    now_outputs = target_model(Test_img)
print(torch.mean(torch.var(soft_max(now_outputs))))
print(soft_max(now_outputs))
vis.images(Test_img)#脆弱化样本后
pre_label = predresult(target_model,Test_img)
target_model.train()
optimizer = torch.optim.SGD(target_model.parameters(), lr=1e-3)
size = len(train_dataloader.dataset)
for batch, (X, y) in enumerate(train_dataloader):
    X, y = X.to(device), y.to(device)

    # Compute prediction error
    pred = target_model(X)
    cost_f = loss(pred, y)

    # Backpropagation
    optimizer.zero_grad()
    cost_f.backward()
    optimizer.step()

    if batch % 100 == 0:
        cost_f, current = cost_f.item(), batch * len(X)
        print(f"loss: {cost_f:>7f}  [{current:>5d}/{size:>5d}]")
# test_model(test_dataloader,target_model)#测试模型精度
after_label = predresult(target_model,Test_img)
print(pre_label)
print(after_label)