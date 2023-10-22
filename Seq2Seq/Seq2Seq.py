import torch
from torch import nn

from NModule.NModule import Encoder, Decoder


class Seq2SeqEncoder(Encoder):
    # 可以是双向的
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, rnn=nn.GRU, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = rnn(embed_size, num_hiddens, num_layers, dropout=dropout)

    def forward(self, X, *args):
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        X = self.embedding(X)
        # 把batch size换到前面，时间步换在中间
        X = X.permute(1, 0, 2)
        # 每次训练使用默认为0的初始状态
        # 对每个句子都是初始状态
        output, state = self.rnn(X)
        return output, state


class Seq2SeqDecoder(Decoder):
    # 只关心最后一个hidden state， 所以可以处理变长的句子
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, rnn=nn.GRU, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = rnn(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)
        print("RNN Layer:", self.rnn)

    def init_state(self, enc_output, *args):
        return enc_output[1]

    def forward(self, X, state):
        X = self.embedding(X).permute(1, 0, 2)
        # 把encoder的最后的状态重复为一个(时间步, 1, 1)的矩阵
        if isinstance(self.rnn, nn.LSTM):
            context = state[0][-1].repeat(X.shape[0], 1, 1)
        else:
            context = state[-1].repeat(X.shape[0], 1, 1)
        X_and_context = torch.cat((X, context), 2)
        states, state = self.rnn(X_and_context, state)
        # 把batch_size放到前面
        output = self.dense(states).permute(1, 0, 2)
        return output, state


def sequence_mask(X, valid_len, value=0):
    """屏蔽无关项"""
    max_len = X.size(1)
    mask = torch.arange(max_len, dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带屏蔽的softmax"""

    def forward(self, pred, labels, valid_len):
        weights = torch.ones_like(labels)
        weights = sequence_mask(weights, valid_len)
        self.reduction = "none"
        unweighted_loss = super().forward(pred.permute(0, 2, 1), labels)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss

