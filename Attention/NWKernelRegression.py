import torch.nn
from torch import nn


class NWKernelRegression(nn.Module):
    def __init__(self, **kwards):
        super(NWKernelRegression, self).__init__(**kwards)
        self.w = nn.Parameter(torch.rand(1), requires_grad=True)

    def forward(self, q, k, v):
        # queries和attention_weights的形状为(查询个数，“键－值”对个数)
        q = q.repeat_interleave(k.shape[1]).reshape((-1, k.shape[1]))
        self.attention_weight = nn.functional.softmax(-0.5 * ((q-k) * self.w)**2, dim=1)
        return torch.bmm(self.attention_weight.unsqueeze(1), v.unsqueeze(-1)).reshape(-1)








