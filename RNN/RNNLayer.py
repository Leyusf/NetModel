import torch
from torch import nn


class RNNLayer(nn.Module):

    def __init__(self, vocab_size, hidden_size, num_layers=1):
        super().__init__()
        self.bidirectional = None
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.net = nn.Linear(self.vocab_size + self.hidden_size, self.hidden_size)
        if self.num_layers != 1:
            self.deep = nn.Linear(self.hidden_size * 2, self.hidden_size)

    def x2h_layer(self, inputs, state):
        states = []
        for X in inputs:
            data = torch.unsqueeze(X, dim=0)
            data = torch.cat((data, state), dim=2)
            state = torch.tanh(self.net(data))
            states.append(state)
        states = torch.cat(states, dim=0)
        return states, state

    def forward(self, inputs, H):
        all_states, last_state = self.x2h_layer(inputs, torch.unsqueeze(H[0], dim=0))
        # 如果只有1层，直接输出
        if self.num_layers == 1:
            return all_states, last_state

        # 如果是深度RNN则将计算得到的隐状态作为输入多次计算
        last_states = [last_state]
        for state in H[1:]:
            states = []
            state = torch.unsqueeze(state, dim=0)
            for X in all_states:
                data = torch.unsqueeze(X, dim=0)
                data = torch.cat((data, state), dim=2)
                state = torch.tanh(self.deep(data))
                states.append(state)
            last_states.append(states[-1])
            all_states = torch.cat(states, dim=0)
        last_states = torch.cat(last_states, dim=0)
        return all_states, last_states
