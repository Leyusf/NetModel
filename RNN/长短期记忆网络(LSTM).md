LSTM是一种与[[门控循环单元(GRU)|GRU]]类似的[[循环神经网络(RNN)|RNN]]，都具有更好的记忆能力，但是比GRU更复杂。
长短期记忆网络引入了**记忆元（memory cell）**，或简称为**单元（cell）**。 有些文献认为记忆元是隐状态的一种特殊类型， 它们与隐状态具有相同的形状，其设计目的是用于记录附加的信息。
这里有三个门和一个候选记忆单元。

### 输入门(input gate):
用来决定何时将数据读入单元。

$$
I_t = \sigma(X_tW_{xi}+H_{t-1}W_{hi}+b_i)
$$

### 忘记门(forget gate):
用于重置单元的内容。

$$
F_t = \sigma(X_tW_{xf}+H_{t-1}W_{hf}+b_f)
$$

### 输出门(output gate):
用来从单元中输出条目。

$$
O_t=\sigma(X_tW_{xo}+H_{t-1}W_{ho}+b_o)
$$

### 候选记忆单元(candidate memory cell):
它的计算与上面描述的三个门的计算类似， 但是使用tanh函数作为激活函数，使得函数的输出范围为(-1,1)。

$$
\bar C_t = tanh(X_tW_{xc}+H_{t-1}+b_c)
$$

### 记忆元(memory cell):
记忆元代表了具有附加信息的隐状态。
与[[门控循环单元(GRU)|GRU]]不同的是，这里的遗忘门与输入门是独立的，GRU是不独立的，不能同时存在的。
其可以被以下公式计算：

$$
C_t = F_t \odot C_{t-1} + I_t \odot \bar C_t
$$

如果遗忘门始终为1且输入门始终为0， 则过去的记忆元 $C_{t-1}$  将随时间被保存并传递到当前时间步。 引入这种设计是为了缓解梯度消失问题， 并更好地捕获序列中的长距离依赖关系。

### 隐状态(hidden state):
最后，我们需要定义如何计算隐状态， 这就是输出门发挥作用的地方。在长短期记忆网络中，它仅仅是记忆元的tanh的门控版本。 这就确保了 $H_t$ 的值始终在区间(−1,1)内：

$$
H_t = O_t \odot tanh(C_t)
$$

只要输出门接近1，我们就能够有效地将所有记忆信息传递给预测部分， 而对于输出门接近0，我们只保留记忆元内的所有信息，而不需要更新隐状态。

## 模型结构：
![[Pasted image 20230630182938.png]](../images/20230630182938.png)


pytorch的API是：
```
lstm_layer = nn.LSTM(len(vocab), num_hiddens)
```
其性能如下：
![[Pasted image 20230630183859.png]](../images/20230630183859.png)

代码实现：
```
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
```