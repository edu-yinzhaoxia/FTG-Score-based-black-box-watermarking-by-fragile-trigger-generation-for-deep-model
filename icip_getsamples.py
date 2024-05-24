import os.path

import pgan
import math
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import core
import argparse
loss_fn = nn.CrossEntropyLoss()#损失函数
soft_max = nn.Softmax(dim=1)
def train_generater(dataloader, model, optimizerG,target_model, target_label,c_a=150):
    size = len(dataloader)
    num_batches = len(dataloader)
    model.train()
    X= dataloader.to(device)
    # print(X)
    target_model.eval()
    pred = target_model(X)
    soft_pred = soft_max(pred)
    # loss = 1*loss_fn(pred,target_label)+150*torch.mean(torch.var(soft_max(pred)))
    loss = 1*loss_fn(pred,target_label)+c_a*torch.mean(torch.var(soft_max(pred)))
    print("此时生成样本在目标分类模型中的标签为：")
    print(torch.var(soft_max(pred)))
    optimizerG.zero_grad()
    loss.backward()
    optimizerG.step()

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# defaultkey = torch.tensor([0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,
#                   0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,
#                   0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9]).to(device)
defaultkey = torch.tensor([0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,
                  0,1,2,3,4,5,6,7,8,9]).to(device)
# defaultkey = torch.tensor([0]).to(device)
sample_savepath = './fragile_sample/'
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a classifier")
    parser.add_argument('--targetmodel', type=str, help="the model you want to train", choices=['resnet18','mlp'],
                        default='resnet18')
    parser.add_argument('--sample_num', type=int, help="the number of triggers you want", default= 2)
    # parser.add_argument('--batch_size', type=int, help='the size of each batch', default=128)
    # parser.add_argument('--epochs', type=int, help='the training times', default=70)
    # parser.add_argument('--optimizer', type=str, help="choose sgd or adam", choices=['sgd', 'adam'], default='sgd')

    args = parser.parse_args()
    targetmodel = args.targetmodel
    sample_num = args.sample_num
    ###MNIST
    # target_model = core.models.ResNet(18)
    # target_model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # target_model.to(device)
    # target_model.load_state_dict(torch.load('./tests/experiments/Resnet_Mnist/ckpt_epoch_100.pth'))

    ##CIFAR10
    # target_model = core.models.ResNet(50)
    # target_model.to(device)
    # target_model.load_state_dict(torch.load('./tests/experiments/ResNet-50_CIFAR-10/ckpt_epoch_100.pth'))

    ##CIFAR100
    target_model = core.models.ResNet(50,100)
    target_model.to(device)
    target_model.load_state_dict(torch.load('./tests/experiments/ResNet-50_CIFAR-100/ckpt_epoch_100.pth'))


    ##Tiny
    # target_model = core.models.vgg19_bn(num_classes=200)
    # target_model.to(device)
    # target_model.load_state_dict(torch.load('./tests/experiments/vgg19_bn_TinyImageNet/ckpt_epoch_1.pth'))

    # te_transformer = transforms.Compose([
    #     transforms.ToTensor(),
    #     # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
    # ])

    # test_data = datasets.CIFAR10(
    #     root="./dataset",
    #     train=False,
    #     download=True,
    #     transform=te_transformer
    # )
    # test_set = DataLoader(test_data, batch_size=256)    #load dataset
    # o_test(test_set,target_model)
    #Create PGAN
    g_model = pgan.Generator(3, 512, 32)  # 生成3维数据，输入1*512维，生成图片大小32*32            #load generator
    total_stages = int(math.log2(32 / 4)) + 1
    for i in range(total_stages - 1):
        g_model.grow_network()
        g_model.flush_network()
    g_model = g_model.to(device)
    #set optimizer for PGAN
    optimizerG = torch.optim.Adam(g_model.parameters(), lr=1e-4, betas=(0.5, 0.999))
    #generate latent vector
    fixed_noise = torch.FloatTensor(50, 512).normal_(0.0, 1.0).to(device)
    #start to generate fragile samples
    epoch = 300
    for i in range(epoch):  # start
        fake = g_model(fixed_noise)
        # torch.save(fake, 'img/ablation_data/ablation_b/fake')
        # exit()
        train_generater(fake, g_model, optimizerG, target_model, defaultkey,c_a=4000)
    # my_test(fake, target_model)  # check the label of trigger in target model
    # c = correct_cal(result_pre(fake, target_model), defaultkey)
    target_model.eval()
    print(target_model(fake).argmax(1))
    # torch.save(fake, './samples/mnist/icip_img')
    # torch.save(fake, './samples/cifar10/icip_img')
    torch.save(fake, './samples/cifar100/icip_img')
    # torch.save(fake, './samples/tiny/icip1_img')