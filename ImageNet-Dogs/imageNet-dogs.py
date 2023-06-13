import collections
import math
import os
import shutil

import torch
import torchvision.transforms.v2
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights

from NModule.Ntraining import train_model, drawGraph, reorg_data, read_csv_labels


def main():
    data_dir = "..\\data\\kaggle_dog_tiny"
    labels = read_csv_labels(os.path.join(data_dir, "labels.csv"))
    print("Samples: ", len(labels))
    print("Classes: ", len(set(labels.values())))
    batch_size = 32
    valid_ratio = 0.1
    lr = 0.01
    lr_decay = 0.9
    lr_period = 2
    weight_decay = 1e-4
    epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # reorg_data(data_dir, valid_ratio)
    train_transform = torchvision.transforms.Compose([
        # 生成一个面积为原始图像面积0.08～1倍的小正方形，
        # 然后将其缩放为高度和宽度均为224像素的正方形
        torchvision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])])
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
    ])

    train_ds, train_valid_ds = [torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train_valid_test', folder),
        transform=train_transform) for folder in ['train', 'train_valid']]
    valid_ds, test_ds = [torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train_valid_test', folder),
        transform=test_transform) for folder in ['valid', 'test']]

    train_iter, train_valid_iter = [DataLoader(
        dataset, batch_size, shuffle=True, drop_last=True)
        for dataset in (train_ds, train_valid_ds)]
    valid_iter = DataLoader(valid_ds, batch_size, shuffle=False,
                            drop_last=False)
    # test_iter = DataLoader(test_ds, batch_size, shuffle=False,
    #                        drop_last=False)

    # 使用预训练模型
    net = nn.Sequential()
    net.features = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    net.output_new = nn.Sequential(
        nn.Linear(1000, 256), nn.ReLU(),
        nn.Linear(256, 120)
    )
    for param in net.features.parameters():
        param.requires_grad = False

    # X = torch.randn(size=(1, 3, 224, 224))
    # net.shape(X)

    loss_fn = nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.SGD(net.output_new.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_period, lr_decay)
    train_loss, test_loss, train_acc, test_acc = train_model(net, loss_fn, optimizer, epochs, device, train_iter,
                                                             valid_iter, save_best=True, scheduler=scheduler,
                                                             init=False)
    drawGraph(train_loss, test_loss, train_acc, test_acc)
    torch.save(net, "ResNet.pt")


if __name__ == '__main__':
    main()
