import torch
from torch import nn

from Attention.AttentionScore import AdditiveAttention, ScaledDotProductAttention
from NModule.NModule import Decoder


class AttentionDecoder(Decoder):
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError


# 初始化解码器的状态，需要下面的输入：
# 编码器在所有时间步的最终层隐状态，将作为注意力的键和值；
# 上一时间步的编码器全层隐状态，将作为初始化解码器的隐状态；
# 编码器有效长度（排除在注意力池中填充词元）。
class Seq2SeqAttentionDecoder(AttentionDecoder):

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, rnn=nn.GRU, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        # 使用加性注意力，key，value的大小与RNN输出的应状态一致, 输出也是num_hiddens
        self.attention = AdditiveAttention(num_hiddens, num_hiddens, num_hiddens, dropout)
        # self.attention = ScaledDotProductAttention(dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = rnn(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)
        print("RNN Layer:", self.rnn)

    def init_state(self, enc_output, enc_valid_lens, *args):
        # outputs的形状为(batch_size，num_steps，num_hiddens).
        # hidden_state的形状为(num_layers，batch_size，num_hiddens)
        outputs, hidden_state = enc_output
        return outputs.permute(1, 0, 2), hidden_state, enc_valid_lens

    def forward(self, X, state):
        # state值得是输出，上一个隐状态和有效长度
        # enc_outputs的形状为(batch_size,num_steps,num_hiddens).
        # hidden_state的形状为(num_layers,batch_size,num_hiddens)
        enc_outputs, hidden_state, enc_valid_lens = state
        X = self.embedding(X).permute(1, 0, 2)
        outputs, self._attention_weights = [], []
        for x in X:
            # query的形状为(batch_size,1,num_hiddens)
            if isinstance(self.rnn, nn.LSTM):
                q = torch.unsqueeze(hidden_state[0][-1], dim=1)
            else:
                q = torch.unsqueeze(hidden_state[-1], dim=1)
            context = self.attention(q, enc_outputs, enc_outputs, enc_valid_lens)
            x_and_context = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            out, hidden_state = self.rnn(x_and_context.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        # 全连接层变换后，outputs的形状为 (num_steps,batch_size,vocab_size)
        outputs = self.dense(torch.cat(outputs, dim=0)).permute(1, 0, 2)
        return outputs, [enc_outputs, hidden_state, enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights


