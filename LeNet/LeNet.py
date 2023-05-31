from torch import nn

from NModule.NModule import NModule


class LeNet(NModule):
    def __init__(self, num_channels, num_classes):
        """
        :param num_channels: 输入通道数
        :param num_classes: 输出类别数
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(num_channels, 6, kernel_size=(5, 5), padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=(5, 5)),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, num_classes)
        )



