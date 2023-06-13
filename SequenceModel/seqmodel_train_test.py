import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

from NModule.NDataSet import NDataSet
from NModule.Ntraining import train_seq
from SequenceModel import SequenceModel


def showPlot(x, y, x_name='x', y_name='y', size=(6, 4)):
    plt.figure(figsize=size)
    plt.plot(x, y)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.show()


# 设定序列时间步
T = 1000
time = torch.arange(1, T + 1, dtype=torch.float32)
# 生成序列并添加噪音
y = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
showPlot(time, y, 'time', 'y')
# 设定与多少个前序列相关
k = 16
features = torch.zeros((T - k, k))
for i in range(k):
    features[:, i] = y[i:T - k + i]
labels = y[k:].reshape((-1, 1))

batch_size, epochs, lr = 16, 5, 0.01
n_train = 600
train_data_loader = DataLoader(NDataSet(features[:n_train], labels[:n_train]), batch_size=batch_size, shuffle=False,
                               num_workers=0, drop_last=True)
test_data_loader = DataLoader(NDataSet(features[n_train:], labels[n_train:]), batch_size=batch_size, shuffle=False,
                              num_workers=0, drop_last=True)

net = SequenceModel(k, 1)
# train
optimizer = torch.optim.Adam(net.parameters(), lr)
loss = nn.MSELoss(reduction="none")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_seq(net, loss, optimizer, epochs, train_data_loader, test_data_loader, device)