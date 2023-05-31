import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from AlexNet import AlexNet
from NModule.NDataSet import NDataSet
from NModule.Ntraining import train_model, drawGraph


def main():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor()
    ])

    batch_size = 128
    epochs = 10
    lr = 0.01
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

    train_dataset = NDataSet(train_data.data.float(), train_data.targets, transforms=transform)
    test_dataset = NDataSet(test_data.data.float(), test_data.targets, transforms=transform)

    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True, drop_last=True, num_workers=4)

    net = AlexNet(1, 10)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, amsgrad=True, weight_decay=weight_decay)
    train_loss, test_loss, train_acc, test_acc = train_model(net, loss_fn, optimizer, epochs, device, train_dataloader,
                                                             test_dataloader)
    drawGraph(train_loss, test_loss, train_acc, test_acc)
    torch.save(net, "AlexNet.pt")


if __name__ == '__main__':
    main()
