from torch import nn


class NModule(nn.Module):
    """
    添加了shape方法来获取网络的形状
    """
    def __init__(self):
        super().__init__()
        self.net = None

    def forward(self, images):
        res = self.net(images)
        return res

    def shape(self, data):
        """
        :param data: 测试的数据
        :return: 基于测试数据的网络形状
        """
        for layer in self.net:
            data = layer(data)
            print(layer.__class__.__name__, 'output shape: \t', data.shape)
