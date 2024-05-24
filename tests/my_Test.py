'''
This is the test code of poisoned training under BadNets.
'''


import os.path as osp

import cv2
import torch
import torch.nn as nn
import torchvision
from torchvision.datasets import DatasetFolder
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, ToPILImage, Resize

import core

if __name__ == '__main__':
    # ========== Set global settings ==========
    global_seed = 666
    deterministic = True
    torch.manual_seed(global_seed)
    CUDA_VISIBLE_DEVICES = '1'
    datasets_root_dir = '../datasets'


    # ========== BaselineMNISTNetwork_MNIST_BadNets ==========
    # dataset = torchvision.datasets.MNIST
    #
    # transform_train = Compose([
    #     ToTensor()
    # ])
    # trainset = dataset(datasets_root_dir, train=True, transform=transform_train, download=True)
    #
    # transform_test = Compose([
    #     ToTensor()
    # ])
    # testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)
    #
    # pattern = torch.zeros((28, 28), dtype=torch.uint8)
    # pattern[-3:, -3:] = 255
    # weight = torch.zeros((28, 28), dtype=torch.float32)
    # weight[-3:, -3:] = 1.0
    # test_model = core.models.ResNet(18)
    # test_model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # badnets = core.BadNets(
    #     train_dataset=trainset,
    #     test_dataset=testset,
    #     # model=core.models.BaselineMNISTNetwork(),
    #     model=test_model,
    #     loss=nn.CrossEntropyLoss(),
    #     y_target=1,
    #     poisoned_rate=0.05,
    #     pattern=pattern,
    #     weight=weight,
    #     seed=global_seed,
    #     deterministic=deterministic
    # )
    #
    # # Train Attacked Model (schedule is set by yamengxi)
    # schedule = {
    #     'device': 'cpu',
    #     'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
    #     'GPU_num': 1,
    #
    #     'benign_training': True,
    #     'batch_size': 128,
    #
    #     'lr': 0.1,
    #     'momentum': 0.9,
    #     'weight_decay': 5e-4,
    #     'gamma': 0.1,
    #     'schedule': [75, 90],
    #
    #     'epochs': 100,
    #
    #     'log_iteration_interval': 100,
    #     'test_epoch_interval': 10,
    #     'save_epoch_interval': 10,
    #
    #     'save_dir': 'experiments',
    #     'experiment_name': 'Resnet_Mnist'
    # }
    # print(torch.cuda.device_count())
    # badnets.train(schedule)


    # # ========== ResNet-18_CIFAR-10_BadNets ==========
    # dataset = torchvision.datasets.CIFAR100
    #
    # transform_train = Compose([
    #     RandomHorizontalFlip(),
    #     ToTensor()
    # ])
    # trainset = dataset(datasets_root_dir, train=True, transform=transform_train, download=True)
    #
    # transform_test = Compose([
    #     ToTensor()
    # ])
    # testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)
    #
    # pattern = torch.zeros((32, 32), dtype=torch.uint8)
    # pattern[-3:, -3:] = 255
    # weight = torch.zeros((32, 32), dtype=torch.float32)
    # weight[-3:, -3:] = 1.0
    #
    # badnets = core.BadNets(
    #     train_dataset=trainset,
    #     test_dataset=testset,
    #     model=core.models.ResNet(50,100),
    #     loss=nn.CrossEntropyLoss(),
    #     y_target=1,
    #     poisoned_rate=0.05,
    #     pattern=pattern,
    #     weight=weight,
    #     seed=global_seed,
    #     deterministic=deterministic
    # )
    #
    # # Train Attacked Model (schedule is the same as https://github.com/THUYimingLi/Open-sourced_Dataset_Protection/blob/main/CIFAR/train_watermarked.py)
    # schedule = {
    #     'device': 'GPU',
    #     'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
    #     'GPU_num': 1,
    #
    #     'benign_training': True,
    #     'batch_size': 128,
    #
    #     'lr': 0.1,
    #     'momentum': 0.9,
    #     'weight_decay': 5e-4,
    #     'gamma': 0.1,
    #     'schedule': [75, 90],
    #
    #     'epochs': 100,
    #
    #     'log_iteration_interval': 100,
    #     'test_epoch_interval': 10,
    #     'save_epoch_interval': 10,
    #
    #     'save_dir': 'experiments',
    #     'experiment_name': 'ResNet-50_CIFAR-100_BadNets'
    # }
    # badnets.train(schedule)


    # ========== ResNet-18_GTSRB_BadNets ==========
    # transform_train = Compose([
    #     ToPILImage(),
    #     Resize((32, 32)),
    #     ToTensor()
    # ])
    # trainset = DatasetFolder(
    #     root=osp.join(datasets_root_dir, 'GTSRB', 'train'), # please replace this with path to your training set
    #     loader=cv2.imread,
    #     extensions=('png',),
    #     transform=transform_train,
    #     target_transform=None,
    #     is_valid_file=None)
    #
    # transform_test = Compose([
    #     ToPILImage(),
    #     Resize((32, 32)),
    #     ToTensor()
    # ])
    # testset = DatasetFolder(
    #     root=osp.join(datasets_root_dir, 'GTSRB', 'testset'), # please replace this with path to your test set
    #     loader=cv2.imread,
    #     extensions=('png',),
    #     transform=transform_test,
    #     target_transform=None,
    #     is_valid_file=None)
    #
    #
    # pattern = torch.zeros((32, 32), dtype=torch.uint8)
    # pattern[-3:, -3:] = 255
    # weight = torch.zeros((32, 32), dtype=torch.float32)
    # weight[-3:, -3:] = 1.0
    #
    # badnets = core.BadNets(
    #     train_dataset=trainset,
    #     test_dataset=testset,
    #     model=core.models.ResNet(18, 43),
    #     loss=nn.CrossEntropyLoss(),
    #     y_target=1,
    #     poisoned_rate=0.05,
    #     pattern=pattern,
    #     weight=weight,
    #     poisoned_transform_train_index=2,
    #     poisoned_transform_test_index=2,
    #     seed=global_seed,
    #     deterministic=deterministic
    # )
    #
    # # Train Attacked Model (schedule is the same as https://github.com/THUYimingLi/Open-sourced_Dataset_Protection/blob/main/GTSRB/train_watermarked.py)
    # schedule = {
    #     'device': 'GPU',
    #     'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
    #     'GPU_num': 1,
    #
    #     'benign_training': False,
    #     'batch_size': 128,
    #
    #     'lr': 0.01,
    #     'momentum': 0.9,
    #     'weight_decay': 5e-4,
    #     'gamma': 0.1,
    #     'schedule': [20],
    #
    #     'epochs': 30,
    #
    #     'log_iteration_interval': 100,
    #     'test_epoch_interval': 10,
    #     'save_epoch_interval': 10,
    #
    #     'save_dir': 'experiments',
    #     'experiment_name': 'ResNet-18_GTSRB_BadNets'
    # }
    # badnets.train(schedule)

    # ========== ResNet-101_TinyImageNet_BadNets ==========

    import torchvision.transforms as transforms
    # from torch.utils.data import DataLoader
    # from torch.utils.data import Dataset
    # import os, glob
    # from torchvision.io import read_image
    # from torchvision.io.image import ImageReadMode
    #
    # # vis = visdom.Visdom(env='sen_img')
    # transform_train = Compose([
    #     RandomHorizontalFlip(),
    #     # transforms.Normalize((122.4786, 114.2755, 101.3963), (70.4924, 68.5679, 71.8127)),
    #     # ToTensor()
    # ])
    # transform_test = Compose([
    #     # ToTensor()
    # ])
    # batch_size = 128
    # trainset = core.MyTinyImageNet(root='..\\datasets',is_train = 'train', transform=transform_train)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    #
    # testset = core.MyTinyImageNet(root='..\\datasets', is_train = 'test', transform=transform_test)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)
    # pattern = torch.zeros((32, 32), dtype=torch.uint8)
    # pattern[-3:, -3:] = 255
    # weight = torch.zeros((32, 32), dtype=torch.float32)
    # weight[-3:, -3:] = 1.0
    # target_model = core.models.ResNet(101,200)
    # target_model.load_state_dict(torch.load('./experiments/ResNet-101_TinyImageNet_BadNets_2023-04-09_19_54_03/ckpt_epoch_2.pth'))
    # badnets = core.BadNets(
    #     train_dataset=trainset,
    #     test_dataset=testset,
    #     model=target_model,
    #     loss=nn.CrossEntropyLoss(),
    #     y_target=1,
    #     poisoned_rate=0.05,
    #     pattern=pattern,
    #     weight=weight,
    #     seed=global_seed,
    #     deterministic=deterministic
    # )
    #
    # # Train Attacked Model (schedule is the same as https://github.com/THUYimingLi/Open-sourced_Dataset_Protection/blob/main/CIFAR/train_watermarked.py)
    # schedule = {
    #     'device': 'GPU',
    #     'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
    #     'GPU_num': 1,
    #
    #     'benign_training': True,
    #     'batch_size': 16,
    #
    #     'lr': 0.1,
    #     'momentum': 0.9,
    #     'weight_decay': 5e-4,
    #     'gamma': 0.1,
    #     'schedule': [150, 180],
    #
    #     'epochs': 200,
    #
    #     'log_iteration_interval': 100,
    #     'test_epoch_interval': 2,
    #     'save_epoch_interval': 2,
    #
    #     'save_dir': 'experiments',
    #     'experiment_name': 'ResNet-101_TinyImageNet_BadNets'
    # }
    # badnets.train(schedule)

    ##################vgg19_bn###########################
    from torch.utils.data import DataLoader
    from torch.utils.data import Dataset
    import os, glob
    from torchvision.io import read_image
    from torchvision.io.image import ImageReadMode

    # vis = visdom.Visdom(env='sen_img')
    transform_train = Compose([
        RandomHorizontalFlip(),
        # transforms.Normalize((122.4786, 114.2755, 101.3963), (70.4924, 68.5679, 71.8127)),
        # ToTensor()
    ])
    transform_test = Compose([
        # ToTensor()
    ])
    batch_size = 128
    trainset = core.MyTinyImageNet(root='..\\datasets',is_train = 'train', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = core.MyTinyImageNet(root='..\\datasets', is_train = 'test', transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)
    pattern = torch.zeros((32, 32), dtype=torch.uint8)
    pattern[-3:, -3:] = 255
    weight = torch.zeros((32, 32), dtype=torch.float32)
    weight[-3:, -3:] = 1.0
    model = core.models.vgg19_bn(num_classes=200)
    # model.load_state_dict(torch.load('./experiments/vgg/vgg.pkl')['state_dict'],strict=False )
    model.load_state_dict(torch.load('./experiments/vgg19_bn_TinyImageNet_BadNets_2023-04-11_15_23_27/ckpt_epoch_5.pth'))
    badnets = core.BadNets(
        train_dataset=trainset,
        test_dataset=testset,
        model = model,
        loss=nn.CrossEntropyLoss(),
        y_target=1,
        poisoned_rate=0.05,
        pattern=pattern,
        weight=weight,
        seed=global_seed,
        deterministic=deterministic
    )

    # Train Attacked Model (schedule is the same as https://github.com/THUYimingLi/Open-sourced_Dataset_Protection/blob/main/CIFAR/train_watermarked.py)
    schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,

        'benign_training': True,
        'batch_size': 256,

        'lr': 0.001,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'gamma': 0.1,
        'schedule': [150, 180],

        'epochs': 200,

        'log_iteration_interval': 100,
        'test_epoch_interval': 1,
        'save_epoch_interval': 1,

        'save_dir': 'experiments',
        'experiment_name': 'vgg19_bn_TinyImageNet_BadNets'
    }
    badnets.train(schedule)