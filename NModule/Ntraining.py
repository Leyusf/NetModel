import collections
import math
import os
import shutil
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn


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


class Timer:
    # author: Mu Li
    """Record multiple running times."""

    def __init__(self):
        """Defined in :numref:`sec_minibatch_sgd`"""
        self.times = []

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()


def accuracy(y_hat, y):
    y_hat = torch.argmax(y_hat, dim=1)
    return (y_hat == y.reshape(y_hat.shape, -1)).sum().float() / y.numel()


def evaluate_model(net, loss_fn, device, dataloader):
    """
    :param net: 模型
    :param loss_fn: 损失函数
    :param device: 设备
    :param dataloader: 使用dataloader来降低内存
    :return: 正确率， 损失
    """
    correct = 0.0
    loss = 0.0
    num = 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            loss += loss_fn(y_hat, y).sum()
            y_hat = torch.argmax(y_hat, dim=1)
            correct += (y_hat == y.reshape(y_hat.shape, -1)).sum().float()
            num += y.numel()
    return correct / num, loss / num


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


def train_model(net, loss_fn, optimizer, epochs, device, dataloader, test_dataloader, save_best=False, init=None,
                scheduler=None):
    if init is False:
        pass
    elif not init:
        net.apply(init_weights)
    else:
        net.apply(init)
    """
    没有早停设计
    :param net: 模型
    :param loss_fn: 损失函数
    :param optimizer: 优化器
    :param epochs: 训练轮数
    :param device: 设备
    :param dataloader: 训练数据的dataloader
    :param test_dataloader: 验证数据的dataloader
    :param save_best: 是否保存最优的模型
    :param init: 权重初始化方法
    :return: 训练精度，损失，验证精度，损失
    """
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
    net.to(device)
    timer = Timer()
    best_acc = 0
    print("Training starting")
    for ep in range(1, epochs + 1):
        correct = 0.0
        loss = 0.0
        num = 0
        net.train()
        timer.start()
        for X, y in dataloader:
            optimizer.zero_grad()
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            loss_ep = loss_fn(y_hat, y).sum()
            loss_ep.backward()
            optimizer.step()
            with torch.no_grad():
                y_hat = torch.argmax(y_hat, dim=1)
                correct += (y_hat == y.reshape(y_hat.shape, -1)).sum().float()
                num += y.numel()
                loss += loss_ep
        timer.stop()

        with torch.no_grad():
            acc = correct / num
            loss = loss / num

        if scheduler:
            scheduler.step()
        net.eval()
        test_acc_ep, test_ls = evaluate_model(net, loss_fn, device, test_dataloader)

        train_loss.append(loss.cpu())
        test_loss.append(test_ls.cpu())

        train_acc.append(acc.cpu())
        test_acc.append(test_acc_ep.cpu())

        if save_best and best_acc < test_acc[-1]:
            best_acc = test_acc[-1]
            torch.save(net.state_dict(), "Best_Param.pt")

        print(f'Epoch: {ep}, train loss: {loss:.3f}, test loss: {test_ls:.3f}, train acc: {acc:.3f}, '
              f'test acc: {test_acc_ep:.3f}')
        print(f'{num / timer.times[-1]:.1f} examples/sec '
              f'on {str(device)} total training time:{timer.sum():.1f} sec')

    return train_loss, test_loss, train_acc, test_acc


def drawGraph(train_ls, test_ls, train_acc, test_acc):
    """
    绘制验证集和训练集的损失与精度变换
    :param train_ls: 训练损失
    :param test_ls: 验证损失
    :param train_acc: 训练精度
    :param test_acc: 验证精度
    :return:
    """
    epochs = [i for i in range(1, 1 + len(train_ls))]
    plt.plot(epochs, train_ls, label='train loss')
    plt.plot(epochs, test_ls, label='test loss')
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

    plt.plot(epochs, train_acc, label='train accuracy')
    plt.plot(epochs, test_acc, label='test accuracy')
    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()


def train_object_detection_model(net, loss_fn, optimizer, epochs, device, dataloader, test_dataloader, save_best=False,
                                 init=None, scheduler=None):
    if init is False:
        pass
    elif not init:
        net.apply(init_weights)
    else:
        net.apply(init)
    """
    没有早停设计
    :param net: 模型
    :param loss_fn: 损失函数
    :param optimizer: 优化器
    :param epochs: 训练轮数
    :param device: 设备
    :param dataloader: 训练数据的dataloader
    :param test_dataloader: 验证数据的dataloader
    :param save_best: 是否保存最优的模型
    :param init: 权重初始化方法
    :return: 训练精度，损失，验证精度，损失
    """
    train_loss = []
    test_loss = []
    net.to(device)
    timer = Timer()
    best_loss = np.inf
    print("Training starting")
    for ep in range(1, epochs + 1):
        loss = []
        num = []
        net.train()
        timer.start()
        for X, y in dataloader:
            optimizer.zero_grad()
            X = X.to(device)
            y = y.to(device)
            output = net(X)
            print(loss_fn[0](output[:, 1], y[:, 0]))
            print(loss_fn[1](y[1:], output[:, 1:]))
            loss_ep = loss_fn[0](output[:, 1], y[:, 0]) + loss_fn[1](y[1:], output[:, 1:])
            print(loss_ep)
            loss_ep.backward()
            optimizer.step()
            with torch.no_grad():
                loss.append(loss_ep)
                num.append(y.numel())
        timer.stop()

        with torch.no_grad():
            num = torch.Tensor(num)
            loss = (torch.Tensor(loss) @ num)
            num = num.sum()
            loss = loss / num

        if scheduler:
            scheduler.step()
        net.eval()
        test_ls = evaluate_object_detection_model(net, loss_fn, device, test_dataloader)

        train_loss.append(loss)
        test_loss.append(test_ls)

        if save_best and best_loss > test_loss[-1]:
            best_loss = test_loss[-1]
            torch.save(net.state_dict(), "Best_Param.pt")

        print(f'Epoch: {ep}, train loss: {loss:.3f}, test loss: {test_ls:.3f}')
        print(f'{num / timer.times[-1]:.1f} examples/sec '
              f'on {str(device)} total training time:{timer.sum():.1f} sec')

    return train_loss, test_loss


def evaluate_object_detection_model(net, loss_fn, device, dataloader):
    """
    :param net: 模型
    :param loss_fn: 损失函数
    :param device: 设备
    :param dataloader: 使用dataloader来降低内存
    :return: 正确率， 损失
    """
    loss = []
    num = []
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            label, bbox = net(X)
            loss.append(loss_fn[0](label, y[0]) + loss_fn[1](y[1:], bbox))
            num.append(y.numel())

        num = torch.Tensor(num)
        loss = (torch.Tensor(loss) @ num) / num.sum()
    return loss


def train_seq(net, loss_fn, optimizer, epochs, train_loader, test_loader, device):
    net.apply(init_weights)
    net.to(device)
    for epoch in range(epochs):
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            l = loss_fn(net(X), y)
            l.sum().backward()
            optimizer.step()
        print(f'epoch {epoch + 1}, train loss: {evaluate_model(net, loss_fn, device, train_loader)[1]:f},'
              f' test loss: {evaluate_model(net, loss_fn, device, test_loader)[1]:f}')
