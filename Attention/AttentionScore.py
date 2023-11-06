import math

import torch
from torch import nn

from Seq2Seq import sequence_mask


def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行softmax操作"""
    # X:3D张量，valid_lens:1D或2D张量
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


# 其中查询、键和值的形状为（批量大小，步数或词元序列长度，特征大小）
class AdditiveAttention(nn.Module):
    def __init__(self, key_size, query_size, num_hiddens, dropout):
        super(AdditiveAttention, self).__init__()
        # 最后的输出是一维的
        self.w_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.w_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, valid_lens):
        q = self.w_q(q)
        k = self.w_k(k)
        # 在维度扩展后，
        # q的形状：(batch_size，查询的个数，1，num_hidden)
        # k的形状：(batch_size，1，“键－值”对的个数，num_hiddens)
        # 使用广播方式进行求和
        features = q.unsqueeze(2) + k.unsqueeze(1)
        # self.w_v仅有一个输出，因此从形状中移除最后那个维度。
        # scores的形状：(batch_size，查询的个数，“键-值”对的个数)
        scores = self.w_v(nn.functional.tanh(features)).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # 再计算使用加性注意力评分函数的注意力
        return torch.bmm(self.dropout(self.attention_weights), v)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(self, q, k, v, valid_lens=None):
        d = q.shape[-1]
        # 设置transpose_b=True为了交换keys的最后两个维度
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(d)

        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), v)

