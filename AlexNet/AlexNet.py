from torch import nn

from NModule.NModule import NModule


class AlexNet(NModule):
    def __init__(self, num_channels, num_classes):
        """
        :param num_channels: 输入通道数
        :param num_classes: 输出类别数
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(num_channels, 96, kernel_size=(11, 11), stride=(4, 4), padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=(5, 5), padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=(3, 3), padding=1), nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=(3, 3), padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=(3, 3), padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.Linear(256 * 5 * 5, 4096), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )
