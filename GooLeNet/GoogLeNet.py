import torch
from torch import nn

from NModule.NModule import NModule


class Inception(nn.Module):
    """
    GoogLeNet的并行块
    """
    def __init__(self, in_channels, c1, c2, c3, c4):
        """
        :param in_channels: 输入通道数
        :param c1: 1X1 卷积层的输出通道数
        :param c2: 1X1 和 3X3 卷积层 的输出通道数
        :param c3: 1X1 和 5X5 卷积层 的输出通道数
        :param c4: 1X1 最大池化层 和 3X3 卷积层 的输出通道数
        """
        super(Inception, self).__init__()
        self.p1 = nn.Conv2d(in_channels, c1, kernel_size=(1, 1))
        self.p2 = nn.Sequential(
            nn.Conv2d(in_channels, c2[0], kernel_size=(1, 1)), nn.ReLU(),
            nn.Conv2d(c2[0], c2[1], kernel_size=(3, 3), padding=1), nn.ReLU()
        )
        self.p3 = nn.Sequential(
            nn.Conv2d(in_channels, c3[0], kernel_size=(1, 1)), nn.ReLU(),
            nn.Conv2d(c3[0], c3[1], kernel_size=(5, 5), padding=2), nn.ReLU()
        )
        self.p4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, c4, kernel_size=(1, 1)), nn.ReLU(),
        )

    def forward(self, X):
        return torch.cat((
            self.p1(X),
            self.p2(X),
            self.p3(X),
            self.p4(X),
        ), dim=1)


class GoogleNet(NModule):
    def __init__(self, in_channels, num_classes):
        """
        :param in_channels: 输入通道数
        :param num_classes: 输出通道数
        """
        super().__init__()
        self.net = nn.Sequential(
            # stage 1
            nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=3), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # stage 2
            nn.Conv2d(64, 64, kernel_size=(1, 1)), nn.ReLU(),
            nn.Conv2d(64, 192, kernel_size=(3, 3), padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # stage 3
            Inception(192, 64, (96, 128), (16, 32), 32),
            Inception(256, 128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
            # stage 4
            Inception(480, 192, (96, 208), (16, 48), 64),
            Inception(512, 160, (112, 224), (24, 64), 64),
            Inception(512, 128, (128, 256), (24, 64), 64),
            Inception(512, 112, (144, 288), (32, 64), 64),
            Inception(528, 256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # stage 5
            Inception(832, 256, (160, 320), (32, 128), 128),
            Inception(832, 384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, num_classes)
        )
