import torch
from torch import nn


class LSTMLayer(nn.Module):

    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.bidirectional = None
        self.num_layers = 1
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.forget_gate = nn.Linear(self.vocab_size + self.hidden_size, self.hidden_size)
        self.input_gate = nn.Linear(self.vocab_size + self.hidden_size, self.hidden_size)
        self.candidate_memory = nn.Linear(self.vocab_size + self.hidden_size, self.hidden_size)
        self.output_gate = nn.Linear(self.vocab_size + self.hidden_size, self.hidden_size)

    def forward(self, inputs, state):
        H, C = state
        states = []
        for X in inputs:
            X = torch.unsqueeze(X, dim=0)
            data = torch.cat((X, H), dim=2)
            F_t = torch.sigmoid(self.forget_gate(data))
            I_t = torch.sigmoid(self.input_gate(data))
            # 候选记忆单元
            _C_t = torch.tanh(self.candidate_memory(data))
            O_t = torch.sigmoid(self.output_gate(data))

            C = (C * F_t) + (I_t * _C_t)
            H = torch.tanh(C) * O_t

            states.append(H)
        states = torch.cat(states, dim=0)
        return states, (H, C)
