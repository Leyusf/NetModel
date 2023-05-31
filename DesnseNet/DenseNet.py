import torch
import torch.nn.functional as F
from torch import nn

from NModule.NModule import NModule


class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_conv):
        """
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param num_conv: 卷积层数
        """
        super().__init__()
        layer = []
        for i in range(num_conv):
            layer.append(nn.Sequential(
                nn.BatchNorm2d(out_channels * i + in_channels), nn.ReLU(),
                nn.Conv2d(out_channels * i + in_channels, out_channels, kernel_size=(3, 3), padding=1)
            ))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for block in self.net:
            Y = block(X)
            X = torch.cat((X, Y), dim=1)
        return X


class TransitionBlock:
    def __init__(self, in_channels, out_channels):
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels), nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)),
            nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        )


class DenseNet(NModule):
    def __init__(self, in_channels, num_classes, conv_arch, growth_channels):
        """
        :param in_channels: 输入通道数
        :param num_classes: 输出通道数
        :param conv_arch: 稠密块卷积层的数量
        :param growth_channels: 每次增长的通道数
        """
        super().__init__()
        num_channels = 64
        blocks = []
        for i, num_conv in enumerate(conv_arch):
            blocks.append(DenseBlock(num_channels, growth_channels, num_conv))
            num_channels += num_conv * growth_channels
            if i != len(conv_arch) - 1:
                blocks.append(TransitionBlock(num_channels, num_channels // 2).block)
                num_channels = num_channels // 2

        self.net = nn.Sequential(
            # stage 1
            nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=3),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            *blocks,
            # last stage
            nn.BatchNorm2d(num_channels), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(num_channels, num_classes)
        )
