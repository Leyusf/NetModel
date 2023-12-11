import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from DCGAN import DCGAN
from utils import train_gan


def generate_and_save_images(model, test_input):
    predictions = np.squeeze(model(test_input).cpu().numpy())
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow((predictions[i] + 1) / 2, cmap='gray')
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

    net = DCGAN(noise_size)
    net = net.to(device)

    g_optim = torch.optim.Adam(net.generator.parameters(), lr=lr, weight_decay=weight_decay)
    d_optim = torch.optim.Adam(net.discriminator.parameters(), lr=lr, weight_decay=weight_decay)

    loss_fn = torch.nn.BCELoss()
    gen_loss, dis_loss = train_gan(net, loss_fn, g_optim, d_optim, epochs, train_dataloader, batch_size, noise_size,
                                   pic_size, device, generate_and_save_images)
    plt.plot(gen_loss, label='G_loss')
    plt.plot(dis_loss, label='D_loss')
    plt.show()
    torch.save(net, "DCGAN.pt")


if __name__ == '__main__':
    main()
