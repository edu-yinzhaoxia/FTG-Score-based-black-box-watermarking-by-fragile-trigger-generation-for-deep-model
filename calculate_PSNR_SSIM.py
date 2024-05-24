import cv2
import numpy as np
import math
import math
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import random
import core
# import visdom
import pandas as pd

def psnr(img1, img2):
   mse = np.mean((img1 - img2) ** 2 )
   if mse < 1.0e-10:
      return 100
   return 10 * math.log10(255.0**2/mse)

def ssim(img1, img2):
  C1 = (0.01 * 255)**2
  C2 = (0.03 * 255)**2
  img1 = img1.astype(np.float64)
  img2 = img2.astype(np.float64)
  kernel = cv2.getGaussianKernel(11, 1.5)
  window = np.outer(kernel, kernel.transpose())
  mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5] # valid
  mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
  mu1_sq = mu1**2
  mu2_sq = mu2**2
  mu1_mu2 = mu1 * mu2
  sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
  sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
  sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
  ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                              (sigma1_sq + sigma2_sq + C2))
  return ssim_map.mean()

def calculate_ssim(img1, img2):
  '''calculate SSIM
  the same outputs as MATLAB's
  img1, img2: [0, 255]
  '''
  if not img1.shape == img2.shape:
    raise ValueError('Input images must have the same dimensions.')
  if img1.ndim == 2:
    return ssim(img1, img2)
  elif img1.ndim == 3:
    if img1.shape[2] == 3:
      ssims = []
      for i in range(3):
        ssims.append(ssim(img1, img2))
      return np.array(ssims).mean()
    elif img1.shape[2] == 1:
      return ssim(np.squeeze(img1), np.squeeze(img2))
  else:
    raise ValueError('Wrong input image dimensions.')

# img1 = cv2.imread("Test2_HR.bmp", 0)
# img2 = cv2.imread("Test2_LR2.bmp", 0)
# ss = calculate_ssim(img1, img2)


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

###加载resnet18-MNIST模型
target_model = core.models.ResNet(18)
target_model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
target_model.to(device)
target_model.load_state_dict(torch.load('./tests/experiments/Resnet_Mnist/ckpt_epoch_100.pth'))
###加载resnet50-CIFAR10模型
# target_model = core.models.ResNet(50)
# target_model.to(device)
# target_model.load_state_dict(torch.load('./tests/experiments/ResNet-50_CIFAR-10/ckpt_epoch_100.pth'))
###加载resnet50-CIFAR100模型
# target_model = core.models.ResNet(50,100)
# target_model.to(device)
# target_model.load_state_dict(torch.load('./tests/experiments/ResNet-50_CIFAR-100/ckpt_epoch_100.pth'))
###加载vgg19_bn-TinyImage模型
# target_model = core.models.vgg19_bn(num_classes=200)
# target_model.to(device)
# target_model.load_state_dict(torch.load('./tests/experiments/vgg19_bn_TinyImageNet/ckpt_epoch_1.pth'))
###########################################################
batch_size = 1
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
##选用CIFAR10
# train_data = datasets.CIFAR10(root='./datasets', train=True, download=True, transform=te_transformer)
# test_data = datasets.CIFAR10(root='./datasets', train=False, download=True, transform=te_transformer)
##选用CIFAR100
# train_data = datasets.CIFAR100(root='./datasets', train=True, download=True, transform=te_transformer)
# test_data = datasets.CIFAR100(root='./datasets', train=False, download=True, transform=te_transformer)
##选用MNIST
train_data = datasets.MNIST(root='./datasets', train=True, download=True, transform=te_transformer)
test_data = datasets.MNIST(root='./datasets', train=False, download=True, transform=te_transformer)
##选用Tiny_Imagenet
# train_data = core.MyTinyImageNet(root='.\\datasets', is_train='train', transform=None)
# test_data = core.MyTinyImageNet(root='.\\datasets', is_train='test', transform=None)

##加载数据集
train_dataloader = DataLoader(train_data, batch_size=batch_size,shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size)
#测试模型精度
# test_model(test_dataloader,target_model)
# size = len(train_dataloader.dataset)
#######MNIST专用#############################
flag = 0
for batch, (X,y) in enumerate(train_dataloader):
    X, y = X.to(device), y.to(device)
    print(X.shape)
    print(y.shape)
    Test_img = torch.unsqueeze(X[0], dim=1)
    print(Test_img.shape)
    Test_label = torch.unsqueeze(y[0], dim=0)
    print(Test_label)
    print(Test_label.shape)

    ori_img = X[0].permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype("uint8")
    # imageB = X[1].permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype("uint8")
    break
#############脆弱化样本，记录过程PSNR，SSIM还有LOSS
# vis = visdom.Visdom(env='test')
loss = nn.CrossEntropyLoss()
soft_max = nn.Softmax(dim=1)
Test_img.requires_grad = True
target_model.eval()
ori_label = Test_label
record_psnr = []
record_ssim = []
record_loss = []
record_pre = []
for i in range(10000):
    # if i in [0,4000,8000,9999,10000]:
        # vis.images(Test_img)
    Test_img.requires_grad = True
    outputs = target_model(Test_img)
    Now_img = Test_img.clone()
    re_loss = torch.var(soft_max(outputs)).detach().cpu().numpy()
    record_loss.append(re_loss)
    record_psnr.append(psnr(ori_img,Now_img.squeeze(0).permute(1, 2, 0).mul(255).clamp(0, 255).cpu().detach().numpy().astype("uint8")))
    record_ssim.append(calculate_ssim(ori_img,Now_img.squeeze(0).permute(1, 2, 0).mul(255).clamp(0, 255).cpu().detach().numpy().astype("uint8")))
    cost = -0*loss(outputs, Test_label) - 100*torch.mean(torch.var(soft_max(outputs)))
    print(torch.mean(torch.var(soft_max(outputs))))
    record_pre.append(soft_max(outputs).cpu().detach().numpy())
    # Update adversarial images
    grad = torch.autograd.grad(cost, Test_img,
                               retain_graph=False, create_graph=False)[0]
    eps = 0.0001
    # eps = 1/511
    Test_img = Test_img + eps * grad.sign()
    Test_img = torch.clamp(Test_img, min=0, max=1).detach()
#####MNIST结束

###############Tiny_Image使用开始#############################
# flag = 0
# for batch, (X,y) in enumerate(train_dataloader):
#     X, y = X.float().to(device), y.to(device)
#     print(X.shape)
#     print(y.shape)
#     # Test_img = X
#     Test_img = torch.unsqueeze(X[0], dim=0)
#     print(Test_img.shape)
#     # Test_label = y
#     Test_label = torch.unsqueeze(y[0], dim=0)
#     print(Test_label)
#     print(Test_label.shape)
#
#     ori_img = X[0].permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype("uint8")
#     # imageB = X[1].permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype("uint8")
#     break
#
# vis = visdom.Visdom(env='test')
# loss = nn.CrossEntropyLoss()
# soft_max = nn.Softmax(dim=1)
# Test_img.requires_grad = True
# target_model.eval()
# ori_label = Test_label
# record_psnr = []
# record_ssim = []
# record_loss = []
# record_pre = []
# for i in range(10000):
#     if i in [0,4000,8000,9999,10000]:
#         vis.images(Test_img)
#     Test_img.requires_grad = True
#     outputs = target_model(Test_img)
#     Now_img = Test_img.clone()
#     Ori_img = Test_img.clone()
#     re_loss = torch.var(soft_max(outputs)).detach().cpu().numpy()
#     #记录相关数据
#     record_loss.append(re_loss)
#     record_psnr.append(psnr(ori_img,Now_img.squeeze(0).permute(1, 2, 0).mul(255).clamp(0, 255).cpu().detach().numpy().astype("uint8")))
#     record_ssim.append(calculate_ssim(ori_img,Now_img.squeeze(0).permute(1, 2, 0).mul(255).clamp(0, 255).cpu().detach().numpy().astype("uint8")))
#     cost = -0*loss(outputs, Test_label) - torch.mean(torch.var(soft_max(outputs)))
#     print(torch.mean(torch.var(soft_max(outputs))))
#     record_pre.append(soft_max(outputs).cpu().detach().numpy())
#     # Update adversarial images
#     grad = torch.autograd.grad(cost, Test_img,
#                                retain_graph=False, create_graph=False)[0]
#     # eps = 1/255
#     eps = 0.001
#     # eps = 1/511
#     Test_img = Test_img + eps * grad.sign()
#     # Test_img = torch.clamp(Test_img, min=0, max=1).detach()
#     Test_img = torch.clamp(Test_img, min=0, max=255.).detach()
# vis.images(Test_img)
#########Tiny_Image使用结束#############################

###############Cifar10使用开始#############################
# flag = 0
# for batch, (X,y) in enumerate(train_dataloader):
#     X, y = X.float().to(device), y.to(device)
#     print(X.shape)
#     print(y.shape)
#     # Test_img = X
#     Test_img = torch.unsqueeze(X[0], dim=0)
#     print(Test_img.shape)
#     # Test_label = y
#     Test_label = torch.unsqueeze(y[0], dim=0)
#     print(Test_label)
#     print(Test_label.shape)
#
#     ori_img = X[0].permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype("uint8")
#     # imageB = X[1].permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype("uint8")
#     break
#
# vis = visdom.Visdom(env='test')
# loss = nn.CrossEntropyLoss()
# soft_max = nn.Softmax(dim=1)
# Test_img.requires_grad = True
# target_model.eval()
# ori_label = Test_label
# record_psnr = []
# record_ssim = []
# record_loss = []
# record_pre = []
# vis.images(Test_img)
# for i in range(10000):
#     if i in [0,4000,8000,9999,10000]:
#         vis.images(Test_img)
#     Test_img.requires_grad = True
#     outputs = target_model(Test_img)
#     Now_img = Test_img.clone()
#     Ori_img = Test_img.clone()
#     re_loss = torch.var(soft_max(outputs)).detach().cpu().numpy()
#     #记录相关数据
#     record_loss.append(re_loss)
#     record_psnr.append(psnr(ori_img,Now_img.squeeze(0).permute(1, 2, 0).mul(255).clamp(0, 255).cpu().detach().numpy().astype("uint8")))
#     record_ssim.append(calculate_ssim(ori_img,Now_img.squeeze(0).permute(1, 2, 0).mul(255).clamp(0, 255).cpu().detach().numpy().astype("uint8")))
#     cost = -0*loss(outputs, Test_label) - torch.mean(torch.var(soft_max(outputs)))
#     print(torch.mean(torch.var(soft_max(outputs))))
#     record_pre.append(soft_max(outputs).cpu().detach().numpy())
#     # Update adversarial images
#     grad = torch.autograd.grad(cost, Test_img,
#                                retain_graph=False, create_graph=False)[0]
#     # eps = 1/255
#     eps = 0.00001
#     # eps = 1/511
#     Test_img = Test_img + eps * grad.sign()
#     # Test_img = torch.clamp(Test_img, min=0, max=1).detach()
#     Test_img = torch.clamp(Test_img, min=0, max=1.).detach()
# vis.images(Test_img)
# #########Cifar10使用结束#############################

##记录预测概率分布

##记录数据
result = pd.DataFrame(columns=['psnr','ssim','loss','pre'])
result['psnr'] = record_psnr
result['ssim'] = record_ssim
result['loss'] = record_loss
result['pre'] = record_pre
result.to_pickle("./tests/record/mnisteps1.csv")
# result.to_pickle("./tests/record/cifar100eps1.csv")
# result.to_pickle("./tests/record/cifar100eps1.csv")
# result.to_pickle("./tests/record/Tinyeps1.csv")
# result.to_csv("./tests/record/test.csv",header=False, index=False)

####################################################



















###微调测试
# vis.images(Test_img)#脆弱化样本后
# pre_label = predresult(target_model,Test_img)
# target_model.train()
# optimizer = torch.optim.SGD(target_model.parameters(), lr=1e-3)
# size = len(train_dataloader.dataset)
# for batch, (X, y) in enumerate(train_dataloader):
#     X, y = X.to(device), y.to(device)
#
#     # Compute prediction error
#     pred = target_model(X)
#     cost_f = loss(pred, y)
#
#     # Backpropagation
#     optimizer.zero_grad()
#     cost_f.backward()
#     optimizer.step()
#
#     if batch % 100 == 0:
#         cost_f, current = cost_f.item(), batch * len(X)
#         print(f"loss: {cost_f:>7f}  [{current:>5d}/{size:>5d}]")
# # test_model(test_dataloader,target_model)#测试模型精度
# after_label = predresult(target_model,Test_img)
# print(pre_label)
# print(after_label)