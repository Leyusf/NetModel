import collections
import math
import os
import shutil

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms.v2

from NModule.Ntraining import train_model, drawGraph
from ResNet import ResNet


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


def reorg_cifar10_data(data_dir, valid_ratio):
    labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
    reorg_train_valid(data_dir, labels, valid_ratio)
    reorg_test(data_dir)


def main():
    data_dir = "..\\data\\cifar-10"
    labels = read_csv_labels(os.path.join(data_dir, "trainLabels.csv"))
    print("Samples: ", len(labels))
    print("Classes: ", len(set(labels.values())))
    batch_size = 128
    valid_ratio = 0.1
    lr = 0.05
    weight_decay = 0.0005
    epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # reorg_cifar10_data(data_dir, valid_ratio)
    train_transform = torchvision.transforms.Compose([
        # 在高度和宽度上将图像放大到40像素的正方形
        torchvision.transforms.Resize(40),
        # 随机裁剪出一个高度和宽度均为40像素的正方形图像，
        # 生成一个面积为原始图像面积0.64～1倍的小正方形，
        # 然后将其缩放为高度和宽度均为32像素的正方形
        torchvision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0), ratio=(1.0, 1.0)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                         [0.2023, 0.1994, 0.2010])])
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                         [0.2023, 0.1994, 0.2010])
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
                            drop_last=True)
    # test_iter = DataLoader(test_ds, batch_size, shuffle=False,
    #                        drop_last=False)

    net = ResNet(3, 10)
    X = torch.randn(size=(1, 3, 40, 40))
    net.shape(X)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay)
    train_loss, test_loss, train_acc, test_acc = train_model(net, loss_fn, optimizer, epochs, device, train_iter,
                                                             valid_iter, save_best=True)
    drawGraph(train_loss, test_loss, train_acc, test_acc)
    torch.save(net, "ResNet.pt")


if __name__ == '__main__':
    main()
