import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from LeNet import LeNet
from NModule.NDataSet import NDataSet
from NModule.Ntraining import train_model, drawGraph

transform = transforms.Compose([
    transforms.ToTensor()
])

batch_size = 128
epochs = 20
lr = 0.001
weight_decay = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data = datasets.MNIST(root="../data/",
                            transform=transform,
                            train=True,
                            download=True)
test_data = datasets.MNIST(root="../data/",
                           transform=transform,
                           train=False,
                           download=True)

train_dataset = NDataSet(train_data.data.float(), train_data.targets, (train_data.data.shape[0], 1, train_data.data.shape[1], train_data.data.shape[2]))
test_dataset = NDataSet(test_data.data.float(), test_data.targets, (test_data.data.shape[0], 1, test_data.data.shape[1], test_data.data.shape[2]))

train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True)
test_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True)

net = LeNet(1, 10)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr, amsgrad=True, weight_decay=weight_decay)
train_loss, test_loss, train_acc, test_acc = train_model(net, loss_fn, optimizer, epochs, device, train_dataloader,
                                                         test_dataloader)
drawGraph(train_loss, test_loss, train_acc, test_acc)
torch.save(net, "LeNet.pt")
