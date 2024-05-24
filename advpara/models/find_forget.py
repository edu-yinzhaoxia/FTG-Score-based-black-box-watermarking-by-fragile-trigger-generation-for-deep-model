import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
import numpy as np
import random
import argparse
import torchvision
import resnet, mymlp
import visdom
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



def mytest_model(dataloader, model, loss_fn):
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
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def dataloader(dataname, batch_size):
    # tr_transformer = transforms.Compose([
    #     transforms.RandomCrop(32,padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
    #     transforms.RandomHorizontalFlip(p=0.5),  # 图像一半的概率翻转，一半的概率不翻转
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
    # ])
    tr_transformer = transforms.Compose([
        transforms.ToTensor(),
    ])
    te_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
    ])
    if dataname == 'cifar10':
        training_data = datasets.CIFAR10(root='..\\dataset', train=True, download=True, transform=tr_transformer)
        test_data = datasets.CIFAR10(root='..\\dataset', train=False, download=True, transform=te_transformer)
        train_dataloader = DataLoader(training_data, batch_size=batch_size,shuffle=False)
        test_dataloader = DataLoader(test_data, batch_size=batch_size)
        return train_dataloader, test_dataloader
    elif dataname == 'mnist':
        training_data = datasets.MNIST(root='..\\dataset', train=True, download=True, transform=ToTensor())
        test_data = datasets.MNIST(root='..\\dataset', train=False, download=True, transform=ToTensor())
        train_dataloader = DataLoader(training_data, batch_size=batch_size,shuffle=False)
        test_dataloader = DataLoader(test_data, batch_size=batch_size)
        return train_dataloader, test_dataloader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a classifier")
    parser.add_argument('--modelname', type=str, help="the model you want to train", choices=['resnet18','resnet50','resnet101','resnet152','mlp'],
                        default='resnet18')
    parser.add_argument('--dataname', type=str, help="choose the dataset", choices=['cifar10','mnist'], default='cifar10')
    parser.add_argument('--batch_size', type=int, help='the size of each batch', default=128)
    parser.add_argument('--epochs', type=int, help='the training times', default=70)
    parser.add_argument('--optimizer', type=str, help="choose sgd or adam", choices=['sgd', 'adam'], default='sgd')

    args = parser.parse_args()
    modelname = args.modelname
    dataname = args.dataname
    batch_size = args.batch_size
    epochs = args.epochs
    optimizer = args.optimizer

    setup_seed(88)
    # load the dataset
    traindata, testdata = dataloader(dataname, batch_size)
    # Set the loss function
    loss_fn = nn.CrossEntropyLoss()
    learning_rate = 1e-1
    if modelname == 'resnet18':
        model = torchvision.models.resnet18()
        inchannel = model.fc.in_features
        model.fc = nn.Linear(inchannel, 10)
        model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        model.maxpool = nn.MaxPool2d(1, 1, 0)
        model = model.to(device)
        if optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=0.9)
    elif modelname == 'resnet50':
        model = torchvision.models.resnet50()
        inchannel = model.fc.in_features
        model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        model.maxpool = nn.MaxPool2d(1, 1, 0)
        model.fc = nn.Linear(inchannel, 10)
        model = model.to(device)
        if optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=0.9)
    elif modelname == 'resnet101':
        model = torchvision.models.resnet101()
        model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        model.maxpool = nn.MaxPool2d(1, 1, 0)
        inchannel = model.fc.in_features
        model.fc = nn.Linear(inchannel, 10)
        model = model.to(device)
        if optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=0.9)
    elif modelname == 'resnet152':
        model = torchvision.models.resnet152()
        model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        model.maxpool = nn.MaxPool2d(1, 1, 0)
        inchannel = model.fc.in_features
        model.fc = nn.Linear(inchannel, 10)
        model = model.to(device)
        if optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=0.9)
    elif modelname == 'mlp':
        model = mymlp.MLMLP().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
    # model = trainer(model, epochs, traindata, testdata, loss_fn, optimizer)
    # torch.save(model.state_dict(),"..\\save_model\\" + "forget_" +modelname + dataname + ".pth")
    vis = visdom.Visdom(env='model_1')
    model.load_state_dict(torch.load('../save_model/forget_resnet18cifar10.pth'))
    forget_times = pd.read_csv("./forget_result.csv",header=None,index_col=False)
    # print(forget_times[0][0])
    size = len(traindata.dataset)
    # model.train()
    i = -1
    image_list = torch.zeros([1,3,32,32],dtype=torch.float32).to(device)
    print(image_list.shape)
    softm = nn.Softmax(dim=0)
    model.eval()
    for batch, (X, y) in enumerate(traindata):
        X, y = X.to(device), y.to(device)
        i+=X.shape[0]
        # Compute prediction error
        pred = model(X)
        j = i-X.shape[0]+1;
        for pre_l,img in zip(pred,X):
            if(forget_times[0][j]>40):
                image_list = torch.cat((image_list,torch.unsqueeze(img,dim=0)))
                if(max(softm(pre_l))<0.5):
                    vis.images(img)
                    vis.histogram(softm(pre_l))
            j+=1

