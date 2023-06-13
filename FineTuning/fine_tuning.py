import os

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights

from NModule.Ntraining import train_model, drawGraph


def main():
    print(os.getcwd())
    # imageNet的标准化
    normalize = torchvision.transforms.Normalize(
        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    )

    train_args = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        normalize
    ])
    train_set = torchvision.datasets.ImageFolder(os.path.join('..\\data\\hotdog', 'train'), train_args)

    test_args = torchvision.transforms.Compose([
        torchvision.transforms.Resize([256, 256]),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        normalize])
    test_set = torchvision.datasets.ImageFolder(os.path.join('..\\data\\hotdog', 'test'), test_args)

    batch_size = 64
    epochs = 10
    lr = 1e-4
    weight_decay = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataloader = DataLoader(train_set, batch_size, shuffle=True, drop_last=True, num_workers=4)
    test_dataloader = DataLoader(test_set, batch_size, shuffle=True, drop_last=True, num_workers=4)

    net = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Linear(net.fc.in_features, 2)
    nn.init.xavier_uniform_(net.fc.weight)

    # 多GPU数据并行
    # devices = [torch.device('cuda:0'), torch.device('cuda:1')]
    # net = nn.DataParallel(net, device_ids=devices)

    loss_fn = nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay)
    train_loss, test_loss, train_acc, test_acc = train_model(net, loss_fn, optimizer, epochs, device, train_dataloader,
                                                             test_dataloader, save_best=True)
    drawGraph(train_loss, test_loss, train_acc, test_acc)
    torch.save(net, "ResNet.pt")


if __name__ == '__main__':
    main()
