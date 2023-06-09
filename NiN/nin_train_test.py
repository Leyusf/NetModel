import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from NModule.NDataSet import NDataSet
from NModule.Ntraining import train_model, drawGraph
from NiN import NiNNet


def main():
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ])

    batch_size = 64
    epochs = 10
    lr = 0.1
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

    nin_arch = [(96, (11, 11), 4, 0), (256, (5, 5), 1, 2), (384, (3, 3), 1, 1)]
    net = NiNNet(1, 10, nin_arch)
    X = torch.randn(size=(1, 1, 224, 224))
    net.shape(X)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay)
    train_loss, test_loss, train_acc, test_acc = train_model(net, loss_fn, optimizer, epochs, device, train_dataloader,
                                                             test_dataloader)
    drawGraph(train_loss, test_loss, train_acc, test_acc)
    torch.save(net, "NiN.pt")


if __name__ == '__main__':
    main()
