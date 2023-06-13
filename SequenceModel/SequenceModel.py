from torch import nn

from NModule.NModule import NModule


class SequenceModel(NModule):
    def __init__(self, in_features, num_classes):
        """
        :param in_features: 输入通道数
        :param num_classes: 输出类别数
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 10), nn.ReLU(),
            nn.Linear(10, num_classes)
        )
