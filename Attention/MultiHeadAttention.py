import torch
from torch import nn

from AttentionScore import AdditiveAttention, ScaledDotProductAttention


class MultiHeadAdditiveAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAdditiveAttention, self).__init__()
        self.heads = []
        for i in range(num_heads):
            self.heads.append(AdditiveAttention(key_size, query_size, num_hiddens, dropout))
        self.W_o = nn.Linear(num_hiddens * num_heads, num_hiddens)

    def forward(self, queries, keys, values, valid_lens):
        output = []
        for head in self.heads:
            output.append(head(queries, keys, values, valid_lens))
        output_concat = torch.cat(output, dim=-1)
        return self.W_o(output_concat)


def transpose_qkv(X, num_heads):
    # 将多个头的计算放到一起优化计算
    # 把head的维度和batch size维度并在一起
    # (batch, steps, key_size) => (batch, steps, heads, key_size // (batch*steps*heads))
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    # (batch, steps, heads, key_size // (batch*steps*heads)) => (batch, heads, steps, key_size // (batch*steps*heads))
    X = X.permute(0, 2, 1, 3)
    # 结果是3维的可以直接用于计算Attention
    # (batch, heads, steps, key_size // (batch*steps*heads)) => (batch * heads, steps, key_size // (batch*steps*heads))
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    # 逆转transpose_qkv的函数操作
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = ScaledDotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, q, k, v, valid_lens):
        q = transpose_qkv(self.W_q(q), self.num_heads)
        k = transpose_qkv(self.W_k(k), self.num_heads)
        v = transpose_qkv(self.W_v(v), self.num_heads)
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)
        output = self.attention(q, k, v, valid_lens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


class LoopMultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super(LoopMultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = ScaledDotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens * num_heads, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens * num_heads, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens * num_heads, bias=bias)
        self.W_o = nn.Linear(num_hiddens * num_heads, num_hiddens, bias=bias)

    def forward(self, q, k, v, valid_lens):
        q = self.W_q(q)
        k = self.W_k(k)
        v = self.W_v(v)
        attention = self.attention(q, k, v, valid_lens)
        return self.W_o(attention)
