from torch import nn

from NModule.NModule import NModule


class VGGBlock:
    def __init__(self, num_conv, in_channels, out_channels):
        """
        :param num_conv: 卷积层数量
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        """
        layers = []
        for _ in range(num_conv):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1))
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)


class VGGNet(NModule):
    def __init__(self, in_channels, num_classes, conv_arch, vgg_size):
        """
        :param in_channels: 输入的通道数
        :param conv_arch: 输入的VGG块结构，是一个元组, 例如(1, 64)表示这个VGG块中有一个卷积层，输出通道是64
        :param vgg_size: VGG块输出图片的大小
        :param num_classes: 类别数
        """
        super().__init__()
        vgg_blocks = []
        for (num_conv, out_channels) in conv_arch:
            vgg_blocks.append(VGGBlock(num_conv, in_channels, out_channels).block)
            in_channels = out_channels

        self.net = nn.Sequential(
            *vgg_blocks,
            # 全连接部分
            nn.Flatten(),
            nn.Linear(in_channels * vgg_size * vgg_size, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )
