import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import random
import core
import visdom
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

def purturbation(grad,mean=0,std=0.01):   #修改idx位置处的权重
    grad += torch.normal(mean = mean,std  = std,size=grad.size()).to(device)

def comparelabel(prelabel,afterlabel):
    index = 0
    if type(prelabel=='list'):
        size = len(prelabel)
    else:
        size = prelabel.shape[0]
    for x, y in zip(prelabel, afterlabel):
        if x == y:
            index += 1
    return index / size

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

def test_model_tiny(dataloader, model):
    size = len(dataloader.dataset)
    # num_batches = len(dataloader)
    model.eval()
    correct = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.float().to(device), y.to(device)
            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%\n")

def train_model(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.float().to(device), y.to(device)
        # X, y = X.to(device), y.to(device)
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
        if batch > size/256:
            break

tr_transformer = transforms.Compose([
    # transforms.ToTensor(),
])
te_transformer = transforms.Compose([
    # transforms.ToTensor(),
])



###加载resnet18-MNIST模型
# target_model = core.models.ResNet(18)
# target_model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
# target_model.to(device)
# target_model.load_state_dict(torch.load('./tests/experiments/Resnet_Mnist/ckpt_epoch_100.pth'))
# ###MNist相关数据准备
# random_img = torch.rand(100,1,28,28).to(device)
# icip_img = torch.load('./samples/mnist/icip_img').to(device)
# fragile_img = torch.load('./samples/mnist/all_img').to(device)
# #选用MNIST
# train_data = datasets.MNIST(root='./datasets', train=True, download=True, transform=te_transformer)
# test_data = datasets.MNIST(root='./datasets', train=False, download=True, transform=te_transformer)

###加载resnet50-CIFAR10模型
# target_model = core.models.ResNet(50)
# target_model.to(device)
# target_model.load_state_dict(torch.load('./tests/experiments/ResNet-50_CIFAR-10/ckpt_epoch_100.pth'))
# random_img = torch.rand(100,3,32,32).to(device)
# icip_img = torch.load('./samples/cifar10/icip_img').to(device)
# fragile_img = torch.load('./samples/cifar10/all_img').to(device)
# #选用CIFAR10
# train_data = datasets.CIFAR10(root='./datasets', train=True, download=True, transform=te_transformer)
# test_data = datasets.CIFAR10(root='./datasets', train=False, download=True, transform=te_transformer)

###加载resnet50-CIFAR100模型
# target_model = core.models.ResNet(50,100)
# target_model.to(device)
# target_model.load_state_dict(torch.load('./tests/experiments/ResNet-50_CIFAR-100/ckpt_epoch_100.pth'))
# random_img = torch.rand(100,3,32,32).to(device)
# icip_img = torch.load('./samples/cifar100/icip_img').to(device)
# fragile_img = torch.load('./samples/cifar100/all_img').to(device)
# #选用CIFAR100
# train_data = datasets.CIFAR100(root='./datasets', train=True, download=True, transform=te_transformer)
# test_data = datasets.CIFAR100(root='./datasets', train=False, download=True, transform=te_transformer)

###加载vgg19_bn-TinyImage模型
target_model = core.models.vgg19_bn(num_classes=200)
target_model.to(device)
target_model.load_state_dict(torch.load('./tests/experiments/vgg19_bn_TinyImageNet/ckpt_epoch_1.pth'))
random_img = torch.rand(100,3,64,64).to(device)
icip_img = torch.load('./samples/tiny/icip_img').to(device)
fragile_img = torch.load('./samples/tiny/all_img').to(device)
##选用Tiny_Imagenet
train_data = core.MyTinyImageNet(root='.\\datasets', is_train='train', transform=None)
test_data = core.MyTinyImageNet(root='.\\datasets', is_train='test', transform=None)
###########################################################

##加载数据集
train_dataloader = DataLoader(train_data, batch_size=256,shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=256)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(target_model.parameters(), lr=0.001,momentum=0.9,weight_decay=5e-4)

#####################开始记录测试#####################
# for batch,(X,y) in enumerate(test_dataloader):
#     normal_img = X[0:100].to(device)
#     break
# target_model.eval()
# prelabel_normal = target_model(normal_img).argmax(1)
prelabel_normal = []
target_model.eval()
for batch,(X,y) in enumerate(test_dataloader):
    Prelabel = target_model(X.float().to(device)).argmax(1)
    for pre_i in Prelabel:
        prelabel_normal.append(pre_i.cpu().numpy().astype("uint8"))
prelabel_random = target_model(random_img).argmax(1)
prelabel_icip = target_model(icip_img).argmax(1)
prelabel_fragile = target_model(fragile_img).argmax(1)
# print(prelabel_normal)
print(prelabel_random)
print(prelabel_icip)
print(prelabel_fragile)
# test_model(test_dataloader,target_model) #cifar mnist
test_model_tiny(test_dataloader,target_model)
normal_dectection_rate = []
random_dectection_rate = []
icip_dectection_rate = []
fragile_dectection_rate = []
train_model(train_dataloader,target_model,loss,optimizer)
test_model_tiny(test_dataloader,target_model)
afterlabel_normal = []
target_model.eval()
for batch, (X, y) in enumerate(test_dataloader):
    Aftlabel = target_model(X.float().to(device)).argmax(1)
    for aft_i in Aftlabel:
        afterlabel_normal.append(aft_i.cpu().numpy().astype("uint8"))
afterlabel_random = target_model(random_img).argmax(1)
afterlabel_icip = target_model(icip_img).argmax(1)
afterlabel_fragile = target_model(fragile_img).argmax(1)
print(1.-comparelabel(prelabel_normal,afterlabel_normal))
print(1.-comparelabel(prelabel_random, afterlabel_random))
print(1.-comparelabel(prelabel_icip, afterlabel_icip))
print(1.-comparelabel(prelabel_fragile, afterlabel_fragile))


