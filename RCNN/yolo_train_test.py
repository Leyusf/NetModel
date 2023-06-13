import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from NModule.NDataSet import ObjectDetectDataset
from NModule.Ntraining import drawGraph, train_object_detection_model
from RCNN import YOLO


def main():
    batch_size = 3
    epochs = 10
    lr = 0.05
    weight_decay = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataloader = DataLoader(ObjectDetectDataset("../data/banana-detection"), batch_size, shuffle=True,
                                  drop_last=True, num_workers=4)
    test_dataloader = DataLoader(ObjectDetectDataset("../data/banana-detection", is_train=False), batch_size,
                                 shuffle=True, drop_last=True, num_workers=4)

    net = YOLO(1)
    X = torch.randn(size=(2, 3, 256, 256))
    net.shape(X)

    # 多GPU数据并行
    # devices = [torch.device('cuda:0'), torch.device('cuda:1')]
    # net = nn.DataParallel(net, device_ids=devices)

    loss_dict = [nn.CrossEntropyLoss, nn.L1Loss]
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay)
    train_loss, test_loss, train_acc, test_acc = train_object_detection_model(net, loss_dict, optimizer, epochs, device,
                                                                              train_dataloader,
                                                                              test_dataloader, save_best=True)
    drawGraph(train_loss, test_loss, train_acc, test_acc)
    torch.save(net, "ResNet.pt")


if __name__ == '__main__':
    main()
