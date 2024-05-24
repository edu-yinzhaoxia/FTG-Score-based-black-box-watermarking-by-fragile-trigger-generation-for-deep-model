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
from tqdm import trange
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
# target_model = core.models.ResNet(18)
# target_model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
# target_model.to(device)
# target_model.load_state_dict(torch.load('./tests/experiments/Resnet_Mnist/ckpt_epoch_100.pth'))
###加载resnet50-CIFAR10模型
# target_model = core.models.ResNet(50)
# target_model.to(device)
# target_model.load_state_dict(torch.load('./tests/experiments/ResNet-50_CIFAR-10/ckpt_epoch_100.pth'))
###加载resnet50-CIFAR100模型
# target_model = core.models.ResNet(50,100)
# target_model.to(device)
# target_model.load_state_dict(torch.load('./tests/experiments/ResNet-50_CIFAR-100/ckpt_epoch_100.pth'))
###加载vgg19_bn-TinyImage模型
target_model = core.models.vgg19_bn(num_classes=200)
target_model.to(device)
target_model.load_state_dict(torch.load('./tests/experiments/vgg19_bn_TinyImageNet/ckpt_epoch_1.pth'))
###########################################################
batch_size = 128
setup_seed(66)

tr_transformer = transforms.Compose([
    transforms.ToTensor(),
])
te_transformer = transforms.Compose([
    transforms.ToTensor(),
])

##选用MNIST
# train_data = datasets.MNIST(root='./datasets', train=True, download=True, transform=te_transformer)
# test_data = datasets.MNIST(root='./datasets', train=False, download=True, transform=te_transformer)

##选用CIFAR10
# train_data = datasets.CIFAR10(root='./datasets', train=True, download=True, transform=te_transformer)
# test_data = datasets.CIFAR10(root='./datasets', train=False, download=True, transform=te_transformer)

#选用CIFAR100
# train_data = datasets.CIFAR100(root='./datasets', train=True, download=True, transform=te_transformer)
# test_data = datasets.CIFAR100(root='./datasets', train=False, download=True, transform=te_transformer)

##选用Tiny_Imagenet
train_data = core.MyTinyImageNet(root='.\\datasets', is_train='train', transform=None)
test_data = core.MyTinyImageNet(root='.\\datasets', is_train='test', transform=None)

##加载数据集
train_dataloader = DataLoader(train_data, batch_size=batch_size,shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=batch_size)
#测试模型精度
# test_model(test_dataloader,target_model)
# size = len(train_dataloader.dataset)

#######MNIST专用#############################
# for batch, (X,y) in enumerate(train_dataloader):
#     X, y = X.to(device), y.to(device)
#     this_batch = X
#     break
# idx = 0
# for iii in trange(100):
#     Test_img = torch.unsqueeze(this_batch[idx], dim=1)
#     Test_label = torch.unsqueeze(this_batch[idx], dim=0)
#     #############脆弱化样本，记录过程PSNR，SSIM还有LOSS
#     # vis = visdom.Visdom(env='test')
#     loss = nn.CrossEntropyLoss()
#     soft_max = nn.Softmax(dim=1)
#     Test_img.requires_grad = True
#     target_model.eval()
#     for i in range(3000):
#         Test_img.requires_grad = True
#         outputs = target_model(Test_img)
#         # cost = -0*loss(outputs, Test_label) - 100*torch.mean(torch.var(soft_max(outputs)))
#         cost =  - 100 * torch.mean(torch.var(soft_max(outputs)))
#         # Update adversarial images
#         grad = torch.autograd.grad(cost, Test_img,
#                                    retain_graph=False, create_graph=False)[0]
#         eps = 0.0001
#         # eps = 1/511
#         Test_img = Test_img + eps * grad.sign()
#         Test_img = torch.clamp(Test_img, min=0, max=1).detach()
#     torch.save(Test_img,'./samples/mnist/img'+str(idx))
#     idx+=1
#####MNIST结束

# #######CIFAR10专用#############################
vis = visdom.Visdom(env='test')
for batch, (X,y) in enumerate(train_dataloader):
    # X, y = X.to(device), y.to(device)  #mnist & cifar
    X, y = X.float().to(device), y.to(device)
    this_batch = X
    break
idx = 0
for iii in trange(100):
    Test_img = torch.unsqueeze(this_batch[idx], dim=0)
    Test_label = torch.unsqueeze(this_batch[idx], dim=0)
    #############脆弱化样本，记录过程PSNR，SSIM还有LOSS
    loss = nn.CrossEntropyLoss()
    soft_max = nn.Softmax(dim=1)
    Test_img.requires_grad = True
    target_model.eval()
    vis.images(Test_img)
    for i in range(10000):
        Test_img.requires_grad = True
        outputs = target_model(Test_img)
        # print(torch.mean(torch.var(soft_max(outputs))))
        # cost = -0*loss(outputs, Test_label) - 100*torch.mean(torch.var(soft_max(outputs)))
        cost =  -torch.mean(torch.var(soft_max(outputs)))
        # Update adversarial images
        grad = torch.autograd.grad(cost, Test_img,
                                   retain_graph=False, create_graph=False)[0]
        # eps = 0.00001 #cifar100
        # eps = 0.00001 #cifar10
        eps = 0.001 #tiny & mnist
        Test_img = Test_img + eps * grad.sign()
        # Test_img = torch.clamp(Test_img, min=0, max=1).detach()#cifar & mnist
        Test_img = torch.clamp(Test_img, min=0, max=255.).detach()
    torch.save(Test_img,'./samples/tiny/img'+str(idx))
    idx+=1
    print(torch.var(soft_max(outputs)))
# #####CIFAR10结束

# #######CIFAR100专用#############################
# for batch, (X,y) in enumerate(train_dataloader):
#     X, y = X.to(device), y.to(device)
#     this_batch = X
#     break
# idx = 0
# for iii in trange(100):
#     Test_img = torch.unsqueeze(this_batch[idx], dim=1)
#     Test_label = torch.unsqueeze(this_batch[idx], dim=0)
#     #############脆弱化样本，记录过程PSNR，SSIM还有LOSS
#     # vis = visdom.Visdom(env='test')
#     loss = nn.CrossEntropyLoss()
#     soft_max = nn.Softmax(dim=1)
#     Test_img.requires_grad = True
#     target_model.eval()
#     for i in range(3000):
#         Test_img.requires_grad = True
#         outputs = target_model(Test_img)
#         # cost = -0*loss(outputs, Test_label) - 100*torch.mean(torch.var(soft_max(outputs)))
#         cost =  - 100 * torch.mean(torch.var(soft_max(outputs)))
#         # Update adversarial images
#         grad = torch.autograd.grad(cost, Test_img,
#                                    retain_graph=False, create_graph=False)[0]
#         eps = 0.00001
#         # eps = 1/511
#         Test_img = Test_img + eps * grad.sign()
#         Test_img = torch.clamp(Test_img, min=0, max=1).detach()
#     torch.save(Test_img,'./samples/cifar100/img'+str(idx))
#     idx+=1
# #####CIFAR100结束

# #######Tiny专用#############################
# for batch, (X,y) in enumerate(train_dataloader):
#     X, y = X.to(device), y.to(device)
#     this_batch = X
#     break
# idx = 0
# for iii in trange(100):
#     Test_img = torch.unsqueeze(this_batch[idx], dim=1)
#     Test_label = torch.unsqueeze(this_batch[idx], dim=0)
#     #############脆弱化样本，记录过程PSNR，SSIM还有LOSS
#     # vis = visdom.Visdom(env='test')
#     loss = nn.CrossEntropyLoss()
#     soft_max = nn.Softmax(dim=1)
#     Test_img.requires_grad = True
#     target_model.eval()
#     for i in range(3000):
#         Test_img.requires_grad = True
#         outputs = target_model(Test_img)
#         # cost = -0*loss(outputs, Test_label) - 100*torch.mean(torch.var(soft_max(outputs)))
#         cost =  - 100 * torch.mean(torch.var(soft_max(outputs)))
#         # Update adversarial images
#         grad = torch.autograd.grad(cost, Test_img,
#                                    retain_graph=False, create_graph=False)[0]
#         eps = 0.001
#         # eps = 1/511
#         Test_img = Test_img + eps * grad.sign()
#         Test_img = torch.clamp(Test_img, min=0, max=1).detach()
#     torch.save(Test_img,'./samples/tiny/img'+str(idx))
#     idx+=1
# #####Tiny结束

###############Tiny_Image使用开始#############################
# for batch, (X,y) in enumerate(train_dataloader):
#     X, y = X.float().to(device), y.to(device)
#     Test_img = torch.unsqueeze(X[0], dim=0)
#     Test_label = torch.unsqueeze(y[0], dim=0)
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

####################################################