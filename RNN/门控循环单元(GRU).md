门控循环单元与普通的[[循环神经网络(RNN)|RNN]]之间的关键区别在于： 前者支持隐状态的门控。 这意味着模型有专门的机制来确定应该何时更新隐状态， 以及应该何时重置隐状态。 这些机制是可学习的，并且能够解决了上面列出的问题。 例如，如果第一个词元非常重要， 模型将学会在第一次观测之后不更新隐状态。 同样，模型也可以学会跳过不相关的临时观测。 最后，模型还将学会在需要的时候重置隐状态。 下面我们将详细讨论各类门控。

门控单元可以更加关注于序列中重要的节点。
其结构如下
![[Pasted image 20230626180222.png]](../images/20230626180222.png)
GRU由两个门组成，一个门是重置门(Reset)，一个门是更新门(Update)，分别记为R和Z。
重置门用于**忘记**上一个隐状态，更新门表示当前输入对隐状态由多少的更新量。同时这里还有一个候选隐状态$\hat{H}$。

其公式为：
$$
\begin{split}\begin{aligned}
\mathbf{R}_t = \sigma(\mathbf{X}_t \mathbf{W}_{xr} + \mathbf{H}_{t-1} \mathbf{W}_{hr} + \mathbf{b}_r),\\
\mathbf{Z}_t = \sigma(\mathbf{X}_t \mathbf{W}_{xz} + \mathbf{H}_{t-1} \mathbf{W}_{hz} + \mathbf{b}_z),\\
\tilde{\mathbf{H}}_t = \tanh(\mathbf{X}_t \mathbf{W}_{xh} + \left(\mathbf{R}_t \odot \mathbf{H}_{t-1}\right) \mathbf{W}_{hh} + \mathbf{b}_h),\\
\mathbf{H}_t = \mathbf{Z}_t \odot \mathbf{H}_{t-1}  + (1 - \mathbf{Z}_t) \odot \tilde{\mathbf{H}}_t.
\end{aligned}\end{split}
$$
根据这个公式可知，当$R_t$接近1，$Z_t$接近于0时，等价于RNN。
在pytorch中，调用API：
```
gru_layer = nn.GRU(len(vocab), num_hiddens)
```
其性能如下：
![[Pasted image 20230626181539.png]](../images/20230626181539.png)
其代码实现如下：
```
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
```