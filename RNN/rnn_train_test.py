import torch
from torch import nn

from NModule.Ntraining import train_rnn, predict_rnn
from NModule.textProcessing import load_data
from RNN import RNNModel
from RNNLayer import RNNLayer

batch_size, num_steps = 32, 50
train_iter, vocab = load_data(batch_size, num_steps, "..\\data\\text\\timemachine.txt")
num_hiddens = 256
# rnn_layer = RNNLayer(len(vocab), num_hiddens)
rnn_layer = nn.RNN(len(vocab), num_hiddens)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = RNNModel(rnn_layer, vocab_size=len(vocab))
net = net.to(device)
num_epochs, lr = 500, 1.2
train_rnn(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=True, theta=1)
print(predict_rnn("traveller", 50, net, vocab, device))
torch.save(net, "RNN.pt")
