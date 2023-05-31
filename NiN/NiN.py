from torch import nn

from NModule.NModule import NModule


class NiNBlock:
    def __init__(self, in_channels, out_channels, kernel_size: tuple, stride, padding):
        """
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param kernel_size: 卷积核大小
        :param stride: 步幅
        :param padding: 填充
        """
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1)), nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1)), nn.ReLU(),
        )


class NiNNet(NModule):
    def __init__(self, in_channels, num_classes, conv_arch):
        """
        :param in_channels: 输入通道数
        :param num_classes: 输出通道数
        :param conv_arch: 模型结构
        """
        super().__init__()
        nin_blocks = []
        for (out_channels, kernel_size, stride, padding) in conv_arch:
            nin_blocks.append(NiNBlock(in_channels, out_channels, kernel_size, stride, padding).block)
            nin_blocks.append(nn.MaxPool2d(3, stride=2))
            in_channels = out_channels

        self.net = nn.Sequential(
            *nin_blocks,
            nn.Dropout(0.5),
            NiNBlock(in_channels, num_classes, (3, 3), 1, 1).block,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
