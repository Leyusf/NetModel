import torch
import torch.nn.functional as F
from torch import nn

from NModule.NModule import NModule


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, strides=1, use1x1conv=False):
        """
        :param in_channels: 输入通道
        :param out_channels: 输出通道
        :param strides: 步幅
        :param use1x1conv: 是否使用1x1卷积
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(strides, strides), padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.path = nn.Sequential()
        if use1x1conv:
            self.path = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(strides, strides))

    def forward(self, X):
        Y = self.block(X)
        Y = Y + self.path(X)
        return F.relu(Y)


class ResBlock:
    def __init__(self, in_channels, out_channels, num_residuals, is_first=False):
        """
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param num_residuals: 残差块数量
        :param is_first: 是否是第一个块
        """
        self.block = []
        for i in range(num_residuals):
            if i == 0 and not is_first:
                self.block.append(Residual(in_channels, out_channels, use1x1conv=True, strides=2))
            else:
                self.block.append(Residual(out_channels, out_channels))


class ResNet(NModule):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            # first stage
            nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=3),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # Res Block
            *ResBlock(64, 64, 2, True).block,
            *ResBlock(64, 128, 2).block,
            *ResBlock(128, 256, 2).block,
            *ResBlock(256, 512, 2).block,
            # last stage
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(), nn.Linear(512, num_classes)
        )
