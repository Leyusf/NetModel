import math

import torch
from torch import nn

from AttentionModel import AttentionDecoder
from MultiHeadAttention import MultiHeadAttention
from NModule.NModule import Encoder
from PositionWiseFFN import AddNorm, PositionWiseFFN
from PositionalEncoding import PositionalEncoding


class EncoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape, ffn_input, ffn_hiddens, num_heads, dropout, bias=False,
                 **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_input, ffn_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))


class TransformerEncoder(Encoder):
    def __init__(self, vocab_size, key_size, query_size, value_size, num_hiddens, norm_shape, ffn_input, ffn_hiddens,
                 num_heads, num_layers, dropout, bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block "+str(i), EncoderBlock(key_size, query_size, value_size, num_hiddens, norm_shape,
                                                               ffn_input, ffn_hiddens, num_heads, dropout, bias))

    def forward(self, X, valid_lens):
        # 嵌入值乘以嵌入维度的平方根进行缩放
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X


class DecoderBlock(nn.Module):
    # 定义第i个块
    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape, ffn_input, ffn_hiddens, num_heads, dropout, i,
                 **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        # 解码器注意力
        self.attention1 = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.norm1 = AddNorm(norm_shape, dropout)
        # 编码器-解码器注意力
        self.attention2 = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.norm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_input, ffn_hiddens, num_hiddens)
        self.norm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        # state有3个东西，分别是encoder的输出，encoder的length和上一个输出的值(类似于隐状态)
        enc_outputs, enc_valid_lens = state[0], state[1]
        if state[2][self.i] is None:
            key_values = X
        else:
            # 把前面时刻的数据累积起来
            key_values = torch.cat((state[2][self.i], X), dim=1)
        state[2][self.i] = key_values
        if self.training:
            # 训练时需要遮蔽后面的东西
            batch_size, num_steps, _ = X.shape
            dec_valid_lens = torch.arange(1, num_steps+1, device=X.device).repeat(batch_size, 1)
        else:
            # 预测时不需要
            dec_valid_lens = None

        # 自注意力
        Y = self.norm1(X, self.attention1(X, key_values, key_values, dec_valid_lens))
        # 编码器－解码器注意力
        Z = self.norm2(Y, self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens))
        return self.norm3(Z, self.ffn(Z)), state


class TransformerDecoder(AttentionDecoder):
    def __init__(self, vocab_size, key_size, query_size, values_size, num_hiddens, norm_shape, ffn_input, ffn_hiddens,
                 num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block " + str(i), DecoderBlock(key_size, query_size, values_size, num_hiddens,
                                                                 norm_shape, ffn_input, ffn_hiddens, num_heads, dropout, i))
        self.dense = nn.Linear(num_hiddens, vocab_size)

    @property
    def attention_weights(self):
        return self._attention_weights

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # 解码器自注意力权重
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            # 编码器-解码器自注意力权重
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

