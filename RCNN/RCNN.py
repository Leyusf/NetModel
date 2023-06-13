import torch.nn as nn
import torch.nn.functional as F


# 定义目标检测网络模型
from NModule.NModule import NModule


class YOLO(NModule):
    def __init__(self, num_classes):
        super(YOLO, self).__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)), nn.Flatten(),
            nn.Linear(64 * 32 * 32, 256), nn.ReLU(),
            nn.Linear(256, (num_classes+1) * 5),
        )

    def forward(self, images):
        return super(YOLO, self).forward(images).view(-1, self.num_classes, 5)
