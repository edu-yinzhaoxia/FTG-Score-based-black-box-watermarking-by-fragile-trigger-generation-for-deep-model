from torchvision.models import resnet50, ResNet50_Weights
import torch
import torch.nn as nn
from torchvision import transforms
import os.path
from torchvision import datasets
from torch.utils.data import Dataset
from PIL import Image
import random
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

def caltimes(prelabel,afterlabel):
    index = 1
    if type(prelabel=='list'):
        size = len(prelabel)
    else:
        size = prelabel.shape[0]
    for x, y in zip(prelabel, afterlabel):
        if x == y:
            index += 1
        else:
            break
    return index


def generate_unique_list(start, end, length):
    # 确保列表长度不超过给定范围
    length = min(length, end - start + 1)

    # 生成一个包含指定范围内所有可能值的列表
    all_values = list(range(start, end + 1))

    # 随机选择不重复的值组成新的列表
    unique_values = random.sample(all_values, length)

    return unique_values

def test_model(dataloader, model):
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

target_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device)
# target_model2 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device)
data = "./datasets/imagenet-mini"

traindir = os.path.join(data, 'train')
valdir = os.path.join(data, 'val')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
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
        transforms.ToTensor(),
        normalize,
    ])
batch_size = 128
train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle = False, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle = False, pin_memory=True)

soft_max = nn.Softmax(dim=1)

fragile_sample = torch.load('./samples/mini/all_img').to(device)
fragile_sample = fragile_sample[0:2]
target_model.eval()
# target_model2.eval()
pred1 = target_model(fragile_sample).argmax(1)
# pred2 = target_model2(fragile_sample).argmax(1)
print(pred1)
# test_model(val_loader,target_model)
for _, param in enumerate(target_model.named_parameters()):
    purturbation(target_model.state_dict()[param[0]], std=0.0005)  # 修改参数

pred2 = target_model(fragile_sample).argmax(1)
print(pred2)
# test_model(val_loader,target_model)
print(comparelabel(pred1,pred2))
exit()
# # 示例用法
start = 0
end = 49
length = 50
consume_times = []
for i in range(10000):
    indexes = generate_unique_list(start,end,length)
    pred_f = [pred1[i] for i in indexes]
    pred_a = [pred2[i] for i in indexes]
    consume_times.append(caltimes(pred_f,pred_a))
times_cifar100result = pd.DataFrame(columns=['fragile'])
times_cifar100result['fragile'] = consume_times
times_cifar100result.to_csv("./tests/record/20230630.csv",header=False, index=False)