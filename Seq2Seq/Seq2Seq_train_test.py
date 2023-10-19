import torch

from NModule.NModule import EncoderDecoder
from NModule.Ntraining import train_seq2seq, predict_seq2seq, bleu
from NModule.machineTranslating import load_data_nmt
from Seq2Seq import Seq2SeqEncoder, Seq2SeqDecoder

embed_size, num_hiddens, num_layers, dropout = 64, 64, 2, 0.1
batch_size, num_steps = 64, 20
lr, num_epochs, device = 0.005, 300, torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_iter, src_vocab, tgt_vocab = load_data_nmt("..\\data\\nmt\\fra-eng", batch_size, num_steps)
encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers,
                        dropout)
decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers,
                        dropout)
net = EncoderDecoder(encoder, decoder)
train_seq2seq(net, train_iter, tgt_vocab, lr, num_epochs, device)

engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .', 'I wanna an apple .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .', 'je veux une pomme .']
for eng, fra in zip(engs, fras):
    translation, attention_weight_seq = predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, device)
    print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')

