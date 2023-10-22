import torch
from torch import nn

from Attention.AttentionModel import Seq2SeqAttentionDecoder
from Attention.utils import show_heatmaps
from NModule.NModule import EncoderDecoder
from NModule.Ntraining import train_seq2seq, predict_seq2seq, bleu
from NModule.machineTranslating import load_data_nmt
from Seq2Seq import Seq2SeqEncoder

embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size, num_steps = 64, 10
lr, num_epochs, device = 0.005, 250, torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_iter, src_vocab, tgt_vocab = load_data_nmt("..\\data\\nmt\\fra-eng", batch_size, num_steps)
rnn_layer = nn.LSTM
encoder = Seq2SeqEncoder(
    len(src_vocab), embed_size, num_hiddens, num_layers, dropout, rnn_layer)

decoder = Seq2SeqAttentionDecoder(
    len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout, rnn_layer)

net = EncoderDecoder(encoder, decoder)
train_seq2seq(net, train_iter, tgt_vocab, lr, num_epochs, device, theta=1)

engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, dec_attention_weight_seq = predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, device, True)
    print(f'{eng} => {translation}, ',
          f'bleu {bleu(translation, fra, k=2):.3f}')

attention_weights = torch.cat([step[0][0][0] for step in dec_attention_weight_seq], 0).reshape((
    1, 1, -1, num_steps))

# 加上一个包含序列结束词元
show_heatmaps(attention_weights[:, :, :, :len(engs[-1].split()) + 1].cpu(), xlabel='Key positions',
              ylabel='Query positions')
