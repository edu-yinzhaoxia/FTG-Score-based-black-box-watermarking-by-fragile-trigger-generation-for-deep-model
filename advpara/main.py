import os.path
import pandas as pd
import math
import torch
import torch.nn as nn
from models import resnet
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

# def label_offset(y):

def o_test(dataloader, model):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}% \n")
    return 100*correct

def fine_tune(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        # loss = loss_fn(pred, y) + (now_accuracy-target_accuracy)
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def mytest(dataloader, model, loss_fn, optimizer):
    param_name = []
    now_accuracy = []
    size = len(dataloader.dataset)
    model.eval()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        # loss = loss_fn(pred, y) + (now_accuracy-target_accuracy)
        loss = loss_fn(pred, y)#计算损失
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()#计算梯度
        # optimizer.step()#关掉这个就不会更新参数

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        for _, param in enumerate(model.named_parameters()):
            print(param[0])#名字
            # print(param[1])#参数
            idx = calculate_xyz(param[1].grad)#计算该层梯度最大的权重
            print_xyz(idx,model.state_dict()[param[0]])#打印未修改前该坐标权重值的大小
            # print_xyz(idx,param[1].grad)
            purturbation_xyz(idx,model.state_dict()[param[0]])#修改参数
            # print(param[1].grad.view(-1)[torch.argmax(torch.abs(param[1].grad))])
            print_xyz(idx,model.state_dict()[param[0]])#打印修改后
            param_name.append(param[0])
            now_accuracy.append(o_test(test_dataloader, target_model))
        result_all = pd.DataFrame(columns=['param_name', 'now_accuracy'])
        result_all['param_name'] = param_name
        result_all['now_accuracy'] = now_accuracy
        result_all.to_csv('./result.csv', header=False, index=False)



def calculate_xyz(grad):          #得到最大梯度权重的位置坐标
    test_max = torch.argmax(torch.abs(grad))
    g_size = grad.size()
    if len(g_size) == 4:
        idx_1 = test_max // (g_size[1] * g_size[2] * g_size[3])
        test_max -= (g_size[1] * g_size[2] * g_size[3]) * idx_1
        idx_2 = test_max // (g_size[2] * g_size[3])    # ep.    3/2 = 1.5; 3//2 = 1
        test_max -= (g_size[2] * g_size[3]) * idx_2
        idx_3 = test_max // (g_size[3])
        idx_4 = (test_max - g_size[3] * idx_3)
        return idx_1,idx_2,idx_3,idx_4
    elif len(g_size) == 3:
        idx_1 = test_max // (g_size[1] * g_size[2])
        test_max -= (g_size[1] * g_size[2]) * idx_1
        idx_2 = test_max // (g_size[2])
        test_max -= (g_size[2]) * idx_2
        idx_3 = test_max
        return idx_1, idx_2, idx_3
    elif len(g_size) == 2:
        idx_1 = test_max // (g_size[1])
        test_max -= (g_size[1]) * idx_1
        idx_2 = test_max
        return idx_1, idx_2
    elif len(g_size) == 1:
        idx_1 = test_max
        return idx_1

def print_xyz(idx,grad):         #打印根据坐标idx，输出该权重大小
    gradsize = grad.size()
    if len(gradsize) == 4:
        print(grad[idx[0]][idx[1]][idx[2]][idx[3]])
        return;
    elif len(gradsize)==3:
        print(grad[idx[0]][idx[1]][idx[2]])
        return;
    elif len(gradsize)==2:
        print(grad[idx[0]][idx[1]])
        return;
    elif len(gradsize)==1:
        print(grad[idx])

def purturbation_xyz(idx,grad):   #修改idx位置处的权重
    gradsize = grad.size()
    if len(gradsize) == 4:
        grad[idx[0]][idx[1]][idx[2]][idx[3]]*=5
        return;
    elif len(gradsize)==3:
        grad[idx[0]][idx[1]][idx[2]]*=2
        return;
    elif len(gradsize)==2:
        grad[idx[0]][idx[1]]*=2
        return;
    elif len(gradsize)==1:
        grad[idx]*=2

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
target_model = resnet.resnet18().to(device)
target_model.load_state_dict(torch.load('./save_model/resnet18cifar10.pth'))
batch_size = 100
#数据集准备
tr_transformer = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
    # transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
])
te_transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
])
train_data = datasets.CIFAR10(root='./dataset', train=True, download=True, transform=te_transformer)
test_data = datasets.CIFAR10(root='./dataset', train=False, download=True, transform=te_transformer)
train_dataloader = DataLoader(train_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)
target_model.state_dict()['fc.weight'] = torch.randn(target_model.state_dict()['fc.weight'].size())
target_model.state_dict()['fc.bias'] = torch.randn(target_model.state_dict()['fc.bias'].size())
#打印所有层的名字
# for name,parameters in target_model.named_parameters():
#     print(name,':',parameters.size())
# exit()
# print(target_model.state_dict()['fc.weight'])
# print(target_model.state_dict()['fc.bias'])
# o_test(test_dataloader,target_model) #测试此时模型的精度
#先冻结所有层信息
# for p in target_model.parameters():
#     p.requires_grad = False
# target_model.layer4.requires_grad_(True)
# target_model.avgpool.requires_grad_(True)
# target_model.fc.requires_grad_(True)
# target_model.state_dict()['fc.weight']
# target_model.state_dict()['fc.bias']
# #打印所有层的参数
# for parameters in target_model.parameters():
#     print(parameters)


learning_rate = 1e-2;
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(target_model.parameters(), lr=learning_rate,momentum=0.9)
o_test(test_dataloader, target_model)
mytest(test_dataloader,target_model,loss_fn,optimizer)




