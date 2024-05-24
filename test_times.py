import core
from torchvision.models import resnet50, ResNet50_Weights
import torch
import torch.nn as nn
import pandas as pd
from torchvision import transforms
import os.path
from torchvision import datasets
from torch.utils.data import Dataset
from PIL import Image
import random
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

def purturbation(grad,mean=0,std=0.01):   #修改idx位置处的权重
    grad += torch.normal(mean = mean,std  = std,size=grad.size()).to(device)

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

##加载resnet18-MNIST模型
target_model = core.models.ResNet(18)
target_model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
target_model.to(device)
target_model.load_state_dict(torch.load('./tests/experiments/Resnet_Mnist/ckpt_epoch_100.pth'))
# ##选用MNIST
# te_transformer = transforms.Compose([
#     transforms.ToTensor(),
# ])
# train_data = datasets.MNIST(root='./datasets', train=True, download=True, transform=te_transformer)
# test_data = datasets.MNIST(root='./datasets', train=False, download=True, transform=te_transformer)
# ##加载数据集
# train_dataloader = DataLoader(train_data, batch_size=256,shuffle=True)
# test_dataloader = DataLoader(test_data, batch_size=256)
# fragile_img = torch.load('./samples/mnist/all_img').to(device)
# for batch,(X,y) in enumerate(test_dataloader):
#     normal_img = X[0:100].float().to(device)
#     break
# random_img = torch.rand(100,1,28,28).to(device)
# icip_img = torch.load('./samples/mnist/icip_img').to(device)
soft_max = nn.Softmax(dim=1)
#
# target_model.eval()
# pred1_fragile = target_model(fragile_img).argmax(1)
# pred1_normal = target_model(normal_img).argmax(1)
# pred1_icip = target_model(icip_img).argmax(1)
# pred1_random = target_model(random_img).argmax(1)
# for _, param in enumerate(target_model.named_parameters()):
#     purturbation(target_model.state_dict()[param[0]], std=0.001)  # 修改参数
# target_model.eval()
# pred2_fragile = target_model(fragile_img).argmax(1)
# pred2_normal = target_model(normal_img).argmax(1)
# pred2_icip = target_model(icip_img).argmax(1)
# pred2_random = target_model(random_img).argmax(1)
#
# # 示例用法
# start = 0
# end = 99
# length = 100
# consume_mini_fragile_times = []
# consume_mini_normal_times = []
# consume_mini_random_times = []
# consume_mini_icip_times = []
# for i in range(10000):
#     result = generate_unique_list(start, end, length)
#     indexes = result
#     pred1_fragile = [pred1_fragile[i] for i in indexes]
#     pred1_normal = [pred1_normal[i] for i in indexes]
#     pred1_random = [pred1_random[i] for i in indexes]
#     pred1_icip = [pred1_icip[i] for i in indexes]
#     pred2_normal = [pred2_normal[i] for i in indexes]
#     pred2_random = [pred2_random[i] for i in indexes]
#     pred2_icip = [pred2_icip[i] for i in indexes]
#     pred2_fragile = [pred2_fragile[i] for i in indexes]
#     consume_mini_fragile_times.append(caltimes(pred1_fragile, pred2_fragile))
#     consume_mini_normal_times.append(caltimes(pred1_normal, pred2_normal))
#     consume_mini_random_times.append(caltimes(pred1_random, pred2_random))
#     consume_mini_icip_times.append(caltimes(pred1_icip, pred2_icip))
# times_miniresult = pd.DataFrame(columns=['normal','random','icip','fragile'])
# times_miniresult['normal'] = consume_mini_normal_times
# times_miniresult['random'] = consume_mini_random_times
# times_miniresult['icip'] = consume_mini_icip_times
# times_miniresult['fragile'] = consume_mini_fragile_times
# times_miniresult.to_csv("./tests/record/times_mini_result.csv",header=False, index=False)
# 输出多个指定索引的元素

# # ###加载resnet50-CIFAR10模型###############################################################################
# target_model = core.models.ResNet(50)
# target_model.to(device)
# target_model.load_state_dict(torch.load('./tests/experiments/ResNet-50_CIFAR-10/ckpt_epoch_100.pth'))
# fragile_img = torch.load('./samples/cifar10/all_img').to(device)
# target_model.eval()
# ##选用Cifar10
# te_transformer = transforms.Compose([
#     transforms.ToTensor(),
# ])
# train_data = datasets.CIFAR10(root='./datasets', train=True, download=True, transform=te_transformer)
# test_data = datasets.CIFAR10(root='./datasets', train=False, download=True, transform=te_transformer)
# ##加载数据集
# train_dataloader = DataLoader(train_data, batch_size=256,shuffle=True)
# test_dataloader = DataLoader(test_data, batch_size=256)
# fragile_img = torch.load('./samples/cifar10/all_img').to(device)
# for batch,(X,y) in enumerate(test_dataloader):
#     normal_img = X[0:100].float().to(device)
#     break
# random_img = torch.rand(100,3,32,32).to(device)
# icip_img = torch.load('./samples/cifar10/icip_img').to(device)
# target_model.eval()
# pred1_fragile = target_model(fragile_img).argmax(1)
# pred1_normal = target_model(normal_img).argmax(1)
# pred1_icip = target_model(icip_img).argmax(1)
# pred1_random = target_model(random_img).argmax(1)
# for _, param in enumerate(target_model.named_parameters()):
#     purturbation(target_model.state_dict()[param[0]], std=0.0005)  # 修改参数
# target_model.eval()
# pred2_fragile = target_model(fragile_img).argmax(1)
# pred2_normal = target_model(normal_img).argmax(1)
# pred2_icip = target_model(icip_img).argmax(1)
# pred2_random = target_model(random_img).argmax(1)
# #
# # 示例用法
# start = 0
# end = 99
# length = 100
# consume_cifar10_fragile_times = []
# consume_cifar10_normal_times = []
# consume_cifar10_random_times = []
# consume_cifar10_icip_times = []
# for i in range(10000):
#     result = generate_unique_list(start, end, length)
#     indexes = result
#     pred1_fragile = [pred1_fragile[i] for i in indexes]
#     pred1_normal = [pred1_normal[i] for i in indexes]
#     pred1_random = [pred1_random[i] for i in indexes]
#     pred1_icip = [pred1_icip[i] for i in indexes]
#     pred2_normal = [pred2_normal[i] for i in indexes]
#     pred2_random = [pred2_random[i] for i in indexes]
#     pred2_icip = [pred2_icip[i] for i in indexes]
#     pred2_fragile = [pred2_fragile[i] for i in indexes]
#     consume_cifar10_fragile_times.append(caltimes(pred1_fragile, pred2_fragile))
#     consume_cifar10_normal_times.append(caltimes(pred1_normal, pred2_normal))
#     consume_cifar10_random_times.append(caltimes(pred1_random, pred2_random))
#     consume_cifar10_icip_times.append(caltimes(pred1_icip, pred2_icip))
# times_cifar10result = pd.DataFrame(columns=['normal','random','icip','fragile'])
# times_cifar10result['normal'] = consume_cifar10_normal_times
# times_cifar10result['random'] = consume_cifar10_random_times
# times_cifar10result['icip'] = consume_cifar10_icip_times
# times_cifar10result['fragile'] = consume_cifar10_fragile_times
# times_cifar10result.to_csv("./tests/record/times_cifar10_result.csv",header=False, index=False)
#
#
#
###加载resnet50-CIFAR100模型#############################################################################
target_model = core.models.ResNet(50,100)
target_model.to(device)
target_model.load_state_dict(torch.load('./tests/experiments/ResNet-50_CIFAR-100/ckpt_epoch_100.pth'))
#
##选用MNIST
te_transformer = transforms.Compose([
    transforms.ToTensor(),
])
train_data = datasets.CIFAR100(root='./datasets', train=True, download=True, transform=te_transformer)
test_data = datasets.CIFAR100(root='./datasets', train=False, download=True, transform=te_transformer)
##加载数据集
train_dataloader = DataLoader(train_data, batch_size=256,shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=256)
fragile_img = torch.load('./samples/cifar100/all_img').to(device)
for batch,(X,y) in enumerate(test_dataloader):
    normal_img = X[0:50].float().to(device)
    break
random_img = torch.rand(50,3,32,32).to(device)
icip_img = torch.load('./samples/cifar100/icip_img').to(device)
target_model.eval()
pred1_fragile = target_model(fragile_img).argmax(1)
pred1_normal = target_model(normal_img).argmax(1)
pred1_icip = target_model(icip_img).argmax(1)
pred1_random = target_model(random_img).argmax(1)
test_model(test_dataloader,target_model)
for _, param in enumerate(target_model.named_parameters()):
    purturbation(target_model.state_dict()[param[0]], std=0.0005)  # 修改参数
test_model(test_dataloader,target_model)
target_model.eval()
pred2_fragile = target_model(fragile_img).argmax(1)
pred2_normal = target_model(normal_img).argmax(1)
pred2_icip = target_model(icip_img).argmax(1)
pred2_random = target_model(random_img).argmax(1)
#
# 示例用法
start = 0
end = 49
length = 100
consume_cifar100_fragile_times = []
consume_cifar100_normal_times = []
consume_cifar100_random_times = []
consume_cifar100_icip_times = []
for i in range(10000):
    result = generate_unique_list(start, end, length)
    indexes = result
    pred1_fragile = [pred1_fragile[i] for i in indexes]
    pred1_normal = [pred1_normal[i] for i in indexes]
    pred1_random = [pred1_random[i] for i in indexes]
    pred1_icip = [pred1_icip[i] for i in indexes]
    pred2_normal = [pred2_normal[i] for i in indexes]
    pred2_random = [pred2_random[i] for i in indexes]
    pred2_icip = [pred2_icip[i] for i in indexes]
    pred2_fragile = [pred2_fragile[i] for i in indexes]
    consume_cifar100_fragile_times.append(caltimes(pred1_fragile, pred2_fragile))
    consume_cifar100_normal_times.append(caltimes(pred1_normal, pred2_normal))
    consume_cifar100_random_times.append(caltimes(pred1_random, pred2_random))
    consume_cifar100_icip_times.append(caltimes(pred1_icip, pred2_icip))
times_cifar100result = pd.DataFrame(columns=['normal','random','icip','fragile'])
times_cifar100result['normal'] = consume_cifar100_normal_times
times_cifar100result['random'] = consume_cifar100_random_times
times_cifar100result['icip'] = consume_cifar100_icip_times
times_cifar100result['fragile'] = consume_cifar100_fragile_times
times_cifar100result.to_csv("./tests/record/times_cifar100_result.csv",header=False, index=False)
#
# ###加载vgg19_bn-TinyImage模型########################################################
# target_model = core.models.vgg19_bn(num_classes=200)
# target_model.to(device)
# target_model.load_state_dict(torch.load('./tests/experiments/vgg19_bn_TinyImageNet/ckpt_epoch_1.pth'))
# fragile_img = torch.load('./samples/tiny/all_img').to(device)
#
# target_model.eval()
# pred1 = target_model(fragile_img).argmax(1)
# print(pred1)
# for _, param in enumerate(target_model.named_parameters()):
#     purturbation(target_model.state_dict()[param[0]], std=0.0005)  # 修改参数
# target_model.eval()
# pred2 = target_model(fragile_img).argmax(1)
# print(pred2)
#
# # 示例用法
# start = 0
# end = 99
# length = 100
# consume_tiny_times = []
# for i in range(10000):
#     result = generate_unique_list(start, end, length)
#     indexes = result
#     pred1 = [pred1[i] for i in indexes]
#     pred2 = [pred2[i] for i in indexes]
#     consume_tiny_times.append(caltimes(pred1, pred2))
# times_tinyresult = pd.DataFrame(columns=['times'])
# times_tinyresult['times'] = consume_tiny_times
# times_tinyresult.to_csv("./tests/record/times_tiny_result.csv",header=False, index=False)