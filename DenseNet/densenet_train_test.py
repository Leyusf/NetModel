import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from NModule.Ntraining import train_model, drawGraph
from DenseNet import DenseNet


def main():
    transform = transforms.Compose([
        transforms.Resize(96),
        transforms.ToTensor()
    ])

    batch_size = 64
    epochs = 10
    lr = 0.05
    weight_decay = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = datasets.FashionMNIST(root="../data/",
                                       transform=transform,
                                       train=True,
                                       download=True)

    test_data = datasets.FashionMNIST(root="../data/",
                                      transform=transform,
                                      train=False,
                                      download=True)

    train_dataloader = DataLoader(train_data, batch_size, shuffle=True, drop_last=True, num_workers=4)
    test_dataloader = DataLoader(test_data, batch_size, shuffle=True, drop_last=True, num_workers=4)

    net = DenseNet(1, 10, [4, 4, 4, 4], 32)
    X = torch.randn(size=(1, 1, 96, 96))
    net.shape(X)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay)
    train_loss, test_loss, train_acc, test_acc = train_model(net, loss_fn, optimizer, epochs, device, train_dataloader,
                                                             test_dataloader)

    drawGraph(train_loss, test_loss, train_acc, test_acc)
    torch.save(net, "DenseNet.pt")


if __name__ == '__main__':
    main()
