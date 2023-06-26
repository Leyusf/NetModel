import torch
from torch import nn


class GRULayer(nn.Module):

    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.bidirectional = None
        self.num_layers = 1
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.net = nn.Linear(self.vocab_size + self.hidden_size, self.hidden_size)
        self.reset = nn.Linear(self.vocab_size + self.hidden_size, self.hidden_size)
        self.update = nn.Linear(self.vocab_size + self.hidden_size, self.hidden_size)

    def forward(self, inputs, state):
        states = []
        for X in inputs:
            X = torch.unsqueeze(X, dim=0)
            data = torch.cat((X, state), dim=2)

            R = torch.sigmoid(self.reset(data))
            R = state * R
            data = torch.cat((X, R), dim=2)

            H_t = torch.tanh(self.net(data))

            Z = torch.sigmoid(self.update(data))
            H_p = Z * state
            H_t = (1-Z) * H_t

            state = H_p + H_t
            states.append(state)
        states = torch.cat(states, dim=0)
        return states, state
