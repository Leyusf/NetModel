import torch

from Attention.AttentionScore import AdditiveAttention, ScaledDotProductAttention
from Attention.utils import show_heatmaps

queries, keys = torch.normal(0, 1, (2, 1, 20)), torch.ones((2, 10, 2))
# values的小批量，两个值矩阵是相同的
values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(
    2, 1, 1)

valid_lens = torch.tensor([2, 6])

attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8,
                              dropout=0.1)
attention.eval()
print(attention(queries, keys, values, valid_lens).shape)
show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)), xlabel='Keys', ylabel='Queries')

queries = torch.normal(0, 1, (2, 1, 2))
attention = ScaledDotProductAttention(dropout=0.5)
attention.eval()
print(attention(queries, keys, values, valid_lens).shape)
show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)), xlabel='Keys', ylabel='Queries')

