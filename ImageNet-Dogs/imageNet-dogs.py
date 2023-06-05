import collections
import math
import os
import shutil

import torch
import torchvision.transforms.v2
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights

from NModule.Ntraining import train_model, drawGraph


def read_csv_labels(file):
    """
    :param file: csv文件地址
    :return: 文件名 + 标签
    """
    with open(file, "r") as f:
        lines = f.readlines()[1:]
        tokens = [line.rstrip().split(',') for line in lines]
        return dict(((name, label) for name, label in tokens))


def copy_to_dir(filename, target_dir):
    """将对应的图片放到文件夹中"""
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)


def reorg_train_valid(data_dir, labels, valid_ratio):
    """将验证集从原始的训练集中拆分出来"""
    # 训练数据集中样本最少的类别中的样本数
    n = collections.Counter(labels.values()).most_common()[-1][1]
    # 验证集中每个类别的样本数
    n_valid_per_label = max(1, math.floor(n * valid_ratio))
    label_count = {}
    for train_file in os.listdir(os.path.join(data_dir, 'train')):
        label = labels[train_file.split('.')[0]]
        fname = os.path.join(data_dir, 'train', train_file)
        copy_to_dir(fname, os.path.join(data_dir, 'train_valid_test',
                                        'train_valid', label))
        if label not in label_count or label_count[label] < n_valid_per_label:
            copy_to_dir(fname, os.path.join(data_dir, 'train_valid_test',
                                            'valid', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            copy_to_dir(fname, os.path.join(data_dir, 'train_valid_test',
                                            'train', label))
    return n_valid_per_label


def reorg_test(data_dir):
    """在预测期间整理测试集，以方便读取"""
    for test_file in os.listdir(os.path.join(data_dir, 'test')):
        copy_to_dir(os.path.join(data_dir, 'test', test_file),
                    os.path.join(data_dir, 'train_valid_test', 'test',
                                 'unknown'))


def reorg_data(data_dir, valid_ratio):
    labels = read_csv_labels(os.path.join(data_dir, 'labels.csv'))
    reorg_train_valid(data_dir, labels, valid_ratio)
    reorg_test(data_dir)


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

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.output_new.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_period, lr_decay)
    train_loss, test_loss, train_acc, test_acc = train_model(net, loss_fn, optimizer, epochs, device, train_iter,
                                                             valid_iter, save_best=True, scheduler=scheduler, init=False)
    drawGraph(train_loss, test_loss, train_acc, test_acc)
    torch.save(net, "ResNet.pt")


if __name__ == '__main__':
    main()
