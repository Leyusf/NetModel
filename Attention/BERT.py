import torch
from torch import nn

from Transformer import EncoderBlock


def get_tokens_and_segments(tokens_a, tokens_b=None):
    # 把连个词元数组变成BERT的输入
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    # 第0段
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments


class BERTEncoder(nn.Module):
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_input, ffn_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768, **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(str(i), EncoderBlock(key_size, query_size, value_size, num_hiddens, norm_shape,
                                                      ffn_input, ffn_hiddens, num_heads, dropout, True))
        self.pos_embedding = nn.Parameter(torch.rand(1, max_len, num_hiddens))

    def forward(self, tokens, segments, valid_lens):
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding.data[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X


# 带掩码的语言模型
class MaskLM(nn.Module):
    def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(
            nn.Linear(num_inputs, num_hiddens),
            nn.ReLU(),
            nn.LayerNorm(num_hiddens),
            nn.Linear(num_hiddens, vocab_size)
        )

    def forward(self, X, pred_positions):
        # X是BERT的输出
        num_pred_positions = pred_positions.shape[1]
        # 是那些位置需要预测
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        return self.mlp(masked_X)


class NextSentencePred(nn.Module):
    def __init__(self, num_inputs, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Linear(num_inputs, 2)

    def forward(self, X):
        return self.output(X)


class BERTModel(nn.Module):
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_input, ffn_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768, hid_in_features=768, mlm_in_features=768,
                 nsp_in_features=768):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape, ffn_input, ffn_hiddens, num_heads, num_layers,
                                   dropout, max_len, key_size, query_size, value_size)
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_features)
        self.hidden = nn.Sequential(
            nn.Linear(hid_in_features, num_hiddens),
            nn.Tanh()
        )

    def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
        encoderd_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoderd_X, pred_positions)
        else:
            mlm_Y_hat = None
        # 用于下一句预测的多层感知机分类器的隐藏层，0是“<cls>”标记的索引
        nsp_Y_hat = self.nsp(self.hidden(encoderd_X[:, 0, :]))
        return encoderd_X, mlm_Y_hat, nsp_Y_hat

