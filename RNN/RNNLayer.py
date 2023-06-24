import torch
from torch import nn


class RNNLayer(nn.Module):

    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.bidirectional = None
        self.num_layers = 1
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.net = nn.Linear(self.vocab_size + self.hidden_size, self.hidden_size)

    def forward(self, inputs, state):
        states = []
        for X in inputs:
            data = torch.unsqueeze(X, dim=0)
            data = torch.cat((data, state), dim=2)
            state = torch.tanh(self.net(data))
            states.append(state)
        states = torch.cat(states, dim=0)
        return states, state
