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
import visdom
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import trange

toPIL = transforms.ToPILImage()

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])

# New weights with accuracy 80.858%
target_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device)

# Best available weights (currently alias for IMAGENET1K_V2)
# Note that these weights may change across versions
# resnet50(weights=ResNet50_Weights.DEFAULT)

# Strings are also supported
# resnet50(weights="IMAGENET1K_V2")

# No weights - random initialization
# resnet50(weights=None)


data = "./datasets/imagenet-mini"

traindir = os.path.join(data, 'train')
valdir = os.path.join(data, 'val')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # normalize,
    ]))

val_dataset = datasets.ImageFolder(
    valdir,
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))
val_trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        normalize,
    ])
batch_size = 1
train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle = True, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle = False, pin_memory=True)


# #-----Test Accuracy-----#
# size = len(val_loader.dataset)
# # num_batches = len(dataloader)
# target_model.eval()
# correct = 0
# with torch.no_grad():
#     for X, y in val_loader:
#         X, y = X.to(device), y.to(device)
#         pred = target_model(X)
#         correct += (pred.argmax(1) == y).type(torch.float).sum().item()
# correct /= size
# print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%\n")

# #######CIFAR10专用#############################
vis = visdom.Visdom(env='test')
# for batch, (X,y) in enumerate(train_loader):
#     # X, y = X.to(device), y.to(device)  #mnist & cifar
#     X, y = X.float().to(device), y.to(device)
#     this_batch = X
#     break
idx = 0
# for iii in trange(1):
for batch, (X,y) in enumerate(train_loader):
    # Test_img = torch.unsqueeze(this_batch, dim=0)
    Test_img = X.to(device)
    # Test_label = torch.unsqueeze(this_batch, dim=0)
    Test_label = y.to(device)
    #############脆弱化样本，记录过程PSNR，SSIM还有LOSS
    loss = nn.CrossEntropyLoss()
    soft_max = nn.Softmax(dim=1)
    Test_img.requires_grad = True
    target_model.eval()
    for i in trange(400):
        # if i in [0,999,1999,2999,3999]:
        if i in [0, 10, 200, 300, 399]:
            vis.images(Test_img)
        Test_img.requires_grad = True
        outputs = target_model(val_trans(Test_img))

        cost =  - torch.mean(torch.var(soft_max(outputs)))
        # print("\n")
        # print(-cost)
        # Update adversarial images
        grad = torch.autograd.grad(cost, Test_img,
                                   retain_graph=False, create_graph=False)[0]
        # eps = 0.00001 #cifar100
        # eps = 0.00001 #cifar10
        # eps = 0.0001 #tiny & mnist
        eps = 1 / 255
        Test_img = Test_img + eps * grad.sign()
        # Test_img = torch.clamp(Test_img, min=0, max=1).detach()#cifar & mnist
        Test_img = torch.clamp(Test_img, min=0, max=1.).detach()
        print(-cost)
    print(-cost)
    print(Test_img)
    final_img = toPIL(torch.squeeze(Test_img,dim=0))
    final_img.save('./samples/mini/img'+str(idx)+'.jpg')
    torch.save(Test_img,'./samples/mini/img'+str(idx))
    idx+=1
    if idx >= 50:
        exit()
    print(torch.var(soft_max(outputs)))
# #####CIFAR10结束



