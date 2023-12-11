import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from CGAN import CGAN


def generate_and_save_images(model, test_input):
    predictions = np.squeeze(model(test_input, torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.int64).cuda())
                             .cpu().numpy())
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow((predictions[i] + 1) / 2, cmap="gray")
        plt.axis('off')
    plt.show()


def train_gan(net, loss_fn, g_optim, d_optim, epochs, train_loader, batch_size, noise_size, device, draw_func):
    gen_loss = []
    dis_loss = []
    count = len(train_loader)
    for e in range(1, epochs + 1):
        d_epoch_loss = 0
        g_epoch_loss = 0
        for img, label in train_loader:
            img = img.to(device)
            label = label.to(device)
            fake = 10 * torch.ones_like(label, device=device)

            noise = torch.randn(batch_size, noise_size, device=device)
            real_output, d_fake_output, g_fake_output = net(img, noise, label)
            # 优化生成器
            g_optim.zero_grad()
            g_loss = loss_fn(g_fake_output, label)
            g_loss.backward()
            g_optim.step()

            # 优化判别器
            d_optim.zero_grad()
            real_loss = loss_fn(real_output, label)
            fake_loss = loss_fn(d_fake_output, fake)
            real_loss.backward()
            fake_loss.backward()

            d_loss = real_loss + fake_loss
            d_optim.step()

            with torch.no_grad():
                d_epoch_loss += d_loss
                g_epoch_loss += g_loss

        with torch.no_grad():
            d_epoch_loss /= count
            g_epoch_loss /= count
            dis_loss.append(d_epoch_loss.item())
            gen_loss.append(g_epoch_loss.item())
            print(f'Epoch: {e}, generator loss: {d_epoch_loss:.3f}, discriminator loss: {g_epoch_loss:.3f}')
            draw_func(net.generator, torch.randn(10, noise_size, device=device))
    return gen_loss, dis_loss


def main():
    batch_size = 512
    epochs = 50
    lr = 0.001
    weight_decay = 0.01
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

    net = CGAN(noise_size, 10, pic_size, 1)
    net = net.to(device)

    g_optim = torch.optim.Adam(net.generator.parameters(), lr=lr)
    d_optim = torch.optim.Adam(net.discriminator.parameters(), lr=lr, weight_decay=weight_decay)

    loss_fn = torch.nn.CrossEntropyLoss()
    gen_loss, dis_loss = train_gan(net, loss_fn, g_optim, d_optim, epochs, train_dataloader, batch_size, noise_size,
                                   device, generate_and_save_images)

    plt.plot(gen_loss, label='G_loss')
    plt.plot(dis_loss, label='D_loss')
    plt.show()
    torch.save(net, "CGAN.pt")


if __name__ == '__main__':
    main()
