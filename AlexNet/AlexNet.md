AlexNet首次证明了学习到的特征可以超越手工设计的特征。AlexNet使用了8层卷积神经网络，并以很大的优势赢得了2012年ImageNet图像识别挑战赛。

![Pasted image 20231118134432.png|433](../images/20231118134432.png)

AlexNet和[[LeNet]]的设计理念非常相似，但也存在显著差异。
1. AlexNet比相对较小的LeNet5要深得多。AlexNet由八层组成：五个卷积层、两个全连接隐藏层和一个全连接输出层。
2. AlexNet使用ReLU而不是sigmoid作为其激活函数。