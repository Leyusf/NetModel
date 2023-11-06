import torch

from MultiHeadAttention import MultiHeadAttention, MultiHeadAdditiveAttention, LoopMultiHeadAttention

num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                               num_hiddens, num_heads, 0.5)
attention.eval()
batch_size, num_queries = 2, 4
num_kvpairs, valid_lens = 6, torch.tensor([3, 2])
X = torch.ones((batch_size, num_queries, num_hiddens))
Y = torch.ones((batch_size, num_kvpairs, num_hiddens))

print(attention(X, Y, Y, valid_lens).shape)
print("-" * 10)
attention = MultiHeadAdditiveAttention(num_hiddens, num_hiddens,
                                       num_hiddens, num_heads, 0.5)
attention.eval()
print(attention(X, Y, Y, valid_lens).shape)
print("-" * 10)
attention = LoopMultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                                   num_hiddens, num_heads, 0.5)
attention.eval()
print(attention(X, Y, Y, valid_lens).shape)
