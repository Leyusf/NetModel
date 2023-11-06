from torch import nn


class NModule(nn.Module):
    """
    添加了shape方法来获取网络的形状
    """

    def __init__(self):
        super().__init__()
        self.net = None

    def forward(self, data):
        res = self.net(data)
        return res

    def shape(self, data):
        """
        :param data: 测试的数据
        :return: 基于测试数据的网络形状
        """
        for layer in self.net:
            data = layer(data)
            print(layer.__class__.__name__, 'output shape: \t', data.shape)


class Encoder(nn.Module):
    """编码器-特征提取网络"""

    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError


class Decoder(nn.Module):
    """解码器-分类器"""

    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_output, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError


class EncoderDecoder(nn.Module):
    """编码器-解码器结构"""

    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_output = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_output, *args)
        return self.decoder(dec_X, dec_state)
