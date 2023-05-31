import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn


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
    return (y_hat == y).sum().float() / y.numel()


def evaluate_model(net, loss_fn, device, dataloader):
    """
    :param net: 模型
    :param loss_fn: 损失函数
    :param device: 设备
    :param dataloader: 使用dataloader来降低内存
    :return: 正确率， 损失
    """
    acc = []
    loss = []
    num = []
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            acc.append(accuracy(y_hat, y))
            loss.append(loss_fn(y_hat, y))
            num.append(y.numel())

        num = torch.Tensor(num)
        acc = (torch.Tensor(acc) @ num) / num.sum()
        loss = (torch.Tensor(loss) @ num) / num.sum()
    return acc, loss


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


def train_model(net, loss_fn, optimizer, epochs, device, dataloader, test_dataloader, save_best=False, init=None):
    if not init:
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
        acc = []
        loss = []
        num = []
        net.train()
        timer.start()
        for X, y in dataloader:
            optimizer.zero_grad()
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            loss_ep = loss_fn(y_hat, y)
            loss_ep.backward()
            optimizer.step()
            with torch.no_grad():
                acc.append(accuracy(y_hat, y))
                loss.append(loss_ep)
                num.append(y.numel())
        timer.stop()

        with torch.no_grad():
            num = torch.Tensor(num)
            acc = (torch.Tensor(acc) @ num)
            loss = (torch.Tensor(loss) @ num)
            num = num.sum()
            acc = acc / num
            loss = loss / num

        net.eval()
        test_acc_ep, test_ls = evaluate_model(net, loss_fn, device, test_dataloader)

        train_loss.append(loss)
        test_loss.append(test_ls)

        train_acc.append(acc)
        test_acc.append(test_acc_ep)

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
