import glob

import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms

from pix2pixGAN import Pix2PixGAN
from utils import CMPDataset


def generate_and_save_images(model, test_anno, test_real):
    prediction = model(test_anno).permute(0, 2, 3, 1).detach().cpu().numpy()
    test_anno = test_anno.permute(0, 2, 3, 1).cpu().numpy()
    test_real = test_real.permute(0, 2, 3, 1).cpu().numpy()
    plt.figure(figsize=(10, 10))
    display_list = [test_anno[10], test_real[10], prediction[10]]
    title = ['Input', 'Ground Truth', 'Output']
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        plt.imshow(display_list[i])
        plt.axis('off')  # 坐标系关掉
    plt.show()


def train_gan(net, loss_fn, g_optim, d_optim, epochs, train_loader, batch_size, noise_size, device, draw_func, LAMBDA=7):
    gen_loss = []
    dis_loss = []
    count = len(train_loader)
    real = torch.ones((batch_size, 1, noise_size, noise_size), device=device)
    fake = torch.zeros((batch_size, 1, noise_size, noise_size), device=device)
    imgs_batch, annos_batch = next(iter(train_loader))
    annos_batch = annos_batch.to(device)
    imgs_batch = imgs_batch.to(device)
    l1_loss = torch.nn.L1Loss()
    for e in range(1, epochs + 1):
        d_epoch_loss = 0
        g_epoch_loss = 0
        for img, anno in train_loader:
            img = img.to(device)
            anno = anno.to(device)
            real_output, d_fake_output, g_fake_output, gen_output = net(img, anno)

            # 优化生成器
            g_optim.zero_grad()
            g_loss = loss_fn(g_fake_output, real)

            gen_l1_loss = l1_loss(gen_output, img)
            g_loss += LAMBDA * gen_l1_loss

            g_loss.backward()
            g_optim.step()

            # 优化判别器
            d_optim.zero_grad()
            real_loss = loss_fn(real_output, real)
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
            draw_func(net.generator, annos_batch, imgs_batch)
    return gen_loss, dis_loss


def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256), antialias=True),
        transforms.Normalize(mean=0.5, std=0.5)
    ])
    imgs_path = glob.glob('..\\data\\CMP_facade_DB_extended\\extended\\*.jpg')
    annos_path = glob.glob('..\\data\\CMP_facade_DB_extended\\extended\\*.png')
    dataset = CMPDataset(imgs_path, annos_path, transform)

    batch_size = 32
    epochs = 100
    lr = 2e-3
    weight_decay = 0.01
    noise_size = 60
    gen_channels = 64
    dis_channels = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = DataLoader(dataset, batch_size, drop_last=True, num_workers=4)

    net = Pix2PixGAN(gen_channels, dis_channels)
    net = net.to(device)

    g_optim = torch.optim.Adam(net.generator.parameters(), lr=lr, betas=(0.5, 0.999))
    d_optim = torch.optim.Adam(net.discriminator.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.5, 0.999))

    loss_fn = torch.nn.BCELoss()
    gen_loss, dis_loss = train_gan(net, loss_fn, g_optim, d_optim, epochs, dataloader, batch_size, noise_size,
                                   device, generate_and_save_images, LAMBDA=100)

    plt.plot(gen_loss, label='G_loss')
    plt.plot(dis_loss, label='D_loss')
    plt.show()
    torch.save(net, "Pix2PixGAN.pt")


if __name__ == '__main__':
    main()
