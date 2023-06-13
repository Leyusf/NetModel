import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from NModule.Ntraining import train_model, drawGraph
from ResNet import ResNet


def main():
    transform = transforms.Compose([
        transforms.Resize(224),
        torchvision.transforms.RandomHorizontalFlip(),  # 上下翻转
        torchvision.transforms.RandomVerticalFlip(),  # 左右翻转
        torchvision.transforms.ColorJitter(
            brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),  # 随机改变亮度，对比度，饱和度和色温
        transforms.ToTensor(),
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
                                      transform=transforms.Compose([
                                          transforms.Resize(224),
                                          transforms.ToTensor()
                                      ]),
                                      train=False,
                                      download=True)

    train_dataloader = DataLoader(train_data, batch_size, shuffle=True, drop_last=True, num_workers=4)
    test_dataloader = DataLoader(test_data, batch_size, shuffle=True, drop_last=True, num_workers=4)

    net = ResNet(1, 10)
    X = torch.randn(size=(1, 1, 224, 224))
    net.shape(X)

    # 多GPU数据并行
    # devices = [torch.device('cuda:0'), torch.device('cuda:1')]
    # net = nn.DataParallel(net, device_ids=devices)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay)
    train_loss, test_loss, train_acc, test_acc = train_model(net, loss_fn, optimizer, epochs, device, train_dataloader,
                                                             test_dataloader, save_best=True)
    drawGraph(train_loss, test_loss, train_acc, test_acc)
    torch.save(net, "ResNet.pt")


if __name__ == '__main__':
    main()
