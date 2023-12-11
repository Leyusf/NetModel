import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from GAN import GAN
from utils import train_gan


def gen_img_plot(model, test_input):
    prediction = np.squeeze(model(test_input))
    prediction = prediction.view(-1, 28, 28).detach().cpu().numpy()
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow((prediction[i] + 1)/2)
        plt.axis('off')
    plt.show()


def main():
    batch_size = 256
    epochs = 50
    lr = 0.0001
    weight_decay = 0
    noise_size = 100
    pic_size = 28
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据准备
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    train_data = datasets.MNIST(root="../data/",
                                transform=transform,
                                train=True,
                                download=True)
    train_dataloader = DataLoader(train_data, batch_size, shuffle=True, drop_last=True, num_workers=4)

    net = GAN(noise_size, pic_size, num_hidden=3, hidden_size=256)
    net = net.to(device)

    g_optim = torch.optim.Adam(net.generator.parameters(), lr=lr, weight_decay=weight_decay)
    d_optim = torch.optim.Adam(net.discriminator.parameters(), lr=lr, weight_decay=weight_decay)

    loss_fn = torch.nn.BCELoss()
    train_gan(net, loss_fn, g_optim, d_optim, epochs, train_dataloader, batch_size, noise_size, pic_size, device,
              gen_img_plot)
    torch.save(net, "GAN.pt")


if __name__ == '__main__':
    main()
