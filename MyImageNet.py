import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# from .core.Mydataloader import MyTinyImageNet
import core
from torchvision.models import vgg19, VGG
import torchvision
import torch.nn as nn
import visdom
from torchsummary import summary
import timm
batch_size = 128
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

if __name__ == '__main__':
    # transform = transforms.Normalize((122.4786, 114.2755, 101.3963), (70.4924, 68.5679, 71.8127))
    # transform = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    # vis = visdom.Visdom(env='sen_img')
    # transform_test = transforms.Compose([torchvision.transforms.Resize((64, 64)),
    #                                      ])
    trainset = core.MyTinyImageNet(root='.\\datasets',is_train = 'train', transform=None)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)

    testset = core.MyTinyImageNet(root='.\\datasets', is_train = 'test', transform=None)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    # or any of these variants
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
    # model.conv1= nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=3,
    #                            bias=False)
    # model.fc = nn.Linear(512 * 4, 200)
    model = core.models.vgg19_bn(num_classes=200)
    # model = vgg19(pretrained=False, num_classes = 200)
    # model = timm.create_model('')
    model.load_state_dict(torch.load('./tests/experiments/vgg/vgg.pkl')['state_dict'],strict=False )
    model.to(device)
    print(model)
    size = len(testloader.dataset)

    # num_batches = len(dataloader)
    # model.eval()
    # correct = 0
    # with torch.no_grad():
    #     for batch, (X, y) in enumerate(testloader):
    #         X, y = X.to(device).float(), y.to(device)
    #         # print(X.shape)
    #         # print(y.shape)
    #         # print(X.shape)
    #         # exit()
    #         pred = model(X)
    #         correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    # correct /= size
    # print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%\n")
    from torchsummary import summary
    summary(model,input_size=(3,64,64))
