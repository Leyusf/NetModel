![[Pasted image 20230626162027.png]](../images/20230626162027.png)
循环神经网络可以被视为一个具有两个[[MLP]]的神经网络。
在每一个时间步的计算，输入$X$和隐状态$H$被拼接为一个矩阵，然后这个矩阵经过一个MLP与激活函数得到新的隐状态。
新的隐状态被输入到一个关于输出的MLP即可得到这一时间步的预测值，重复n个时间步。这就是循环神经网络。

其使用的训练数据是经过[[文本预处理]]之后长度相同的字符串。其预测时，可以使用不同长度的输入，预测一定步数的token。

**每一个时间步的MLP是相同的。**

这是一个简单实现的单层单向的RNN层。
在这一层中，输出是每个时间步的隐状态而不是预测值。
```
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
```

我们可以通过隐状态来预测输出。
![[Pasted image 20230626172504.png]](../images/20230626172504.png)