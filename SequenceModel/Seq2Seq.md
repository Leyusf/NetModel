机器翻译中的输入序列和输出序列都是长度可变的。而Seq2Seq模型可以做到接受不同长度的输入并且返回输出。
Seq2Seq使用[[编码器-解码器]]结构。
![[Pasted image 20230918170501.png]](../images/20230918170501.png)
在编码器部分，网络依次接受输入，最后只输出一个最后的隐状态(Hidden state)。
在解码器部分，网络接受第二种语言的输入以及编码器输出的隐状态作为初始状态。
其中`<bos>`表示句子开始，`<eos>`表示句子结束。


## 编码器

编码器可以使用双向循环网络。
编码器由一个RNN网络组成。
```
class Seq2SeqEncoder(Encoder):  
    # 可以是双向的  
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):  
        super(Seq2SeqEncoder, self).__init__(**kwargs)  
        # 嵌入层  
        self.embedding = nn.Embedding(vocab_size, embed_size)  
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)  
  
    def forward(self, X, *args):  
        # 输出'X'的形状：(batch_size,num_steps,embed_size)  
        X = self.embedding(X)  
        # 把batch size换到前面，时间步换在中间  
        X = X.permute(1, 0, 2)  
        # 每次训练使用默认为0的初始状态  
        # 对每个句子都是初始状态，所以不需要输入初始状态，使用默认值。
        output, state = self.rnn(X)  
        return output, state
```


## 解码器

解码器将编码器的最后一个隐状态作为最开始的输入。之后对于每个输入的单词，将其解码器当前的隐状态以及下一个输入词作为输入传递。
在训练时，正确的单词被依次输入，来训练网络。
在与测试，使用预测的单词作为下一个输入的单词完成预测。
```
class Seq2SeqDecoder(Decoder):  
    # 只关心最后一个hidden state， 所以可以处理变长的句子  
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):  
    
        super(Seq2SeqDecoder, self).__init__(**kwargs)  
        self.embedding = nn.Embedding(vocab_size, embed_size)  
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)  
        self.dense = nn.Linear(num_hiddens, vocab_size)  

    def init_state(self, enc_output, *args):  
        return enc_output[1]  
  
    def forward(self, X, state):  
        X = self.embedding(X).permute(1, 0, 2)  
        # 把encoder的最后的状态重复为一个(时间步, 1, 1)的矩阵  
        context = state[-1].repeat(X.shape[0], 1, 1)  
        X_and_context = torch.cat((X, context), 2)  
        states, state = self.rnn(X_and_context, state)  
        # 把batch_size放到前面  
        output = self.dense(states).permute(1, 0, 2)  
        return output, state
```

以上大体结构如下:
![[Pasted image 20230918171338.png]](../images/20230918171338.png)

## 衡量生成序列的好坏——BLEU

$p_n$ 是预测中所有n-gram的精度。
* 例如，序列 A B C D E F 和预测序列 A B B C D, 有 $p_1=4.5, p_2=3/4, p_3=1/3, p_4=0$ 。

* BLEU定义

$$exp(min(0,1 - {len_{label} \over {len_pred}})) {\prod p_n^{1/2^n}}$$

其中 $len_{pred}$ 惩罚过短的预测， $p_n^{1/2^n}$ 长匹配有高权重。

## 屏蔽无效的信息
```
def sequence_mask(X, valid_len, value=0):  
    """屏蔽无关项"""  
    max_len = X.size(1)  
    mask = torch.arange(max_len, dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]  
    X[~mask] = value  
    return X
```
因为句子的长度不一样，但是矩阵的大小是固定的，所以要屏蔽无效的信息。


## 扩展SoftMax函数
```
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):  
    """带屏蔽的softmax"""  
  
    def forward(self, pred, labels, valid_len):  
        weights = torch.ones_like(labels)  
        weights = sequence_mask(weights, valid_len)  
        self.reduction = "none"  
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(  
            pred.permute(0, 2, 1), labels  
        )  
        weighted_loss = (unweighted_loss * weights).mean(dim=1)  
        return weighted_loss
```
填充的值不需要计算损失，只需要计算原始数据的损失，可以在softmax上加上weight，需要信息的weight是1，不需要的是0。

