import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import random
import core
from tqdm import trange
# import visdom

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

##选用Tiny_Imagenet
train_data = core.MyTinyImageNet(root='.\\datasets', is_train='train', transform=None)
test_data = core.MyTinyImageNet(root='.\\datasets', is_train='test', transform=None)

##加载数据集
train_dataloader = DataLoader(train_data, batch_size=batch_size,shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for batch, (X,y) in enumerate(train_dataloader):
    # X, y = X.to(device), y.to(device)  #mnist & cifar
    X, y = X.float().to(device), y.to(device)
    this_batch = X
    break
idx = 0
Test_img = torch.unsqueeze(this_batch[idx], dim=0)
Test_label = torch.unsqueeze(this_batch[idx], dim=0)
#############脆弱化样本，记录过程PSNR，SSIM还有LOSS
loss = nn.CrossEntropyLoss()
soft_max = nn.Softmax(dim=1)
Test_img.requires_grad = True
target_model.eval()

for i in trange(10000):
    Test_img.requires_grad = True
    outputs = target_model(Test_img)
    # print(torch.mean(torch.var(soft_max(outputs))))
    # cost = -0*loss(outputs, Test_label) - 100*torch.mean(torch.var(soft_max(outputs)))
    cost =  - 100 * torch.mean(torch.var(soft_max(outputs)))
    # Update adversarial images
    grad = torch.autograd.grad(cost, Test_img,
                               retain_graph=False, create_graph=False)[0]
    # eps = 0.00001 #cifar100
    # eps = 0.00001 #cifar10
    eps = 0.001 #tiny & mnist
    Test_img = Test_img + eps * grad.sign()
    # Test_img = torch.clamp(Test_img, min=0, max=1).detach()#cifar & mnist
    Test_img = torch.clamp(Test_img, min=0, max=255.).detach()
# torch.save(Test_img,'./samples/tiny/img'+str(idx))
idx+=1
print(torch.var(soft_max(outputs)))