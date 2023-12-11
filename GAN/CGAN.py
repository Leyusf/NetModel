import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, noise_dim, label_dim):
        super(Generator, self).__init__()

        self.label_up_sample = nn.Sequential(
            nn.Embedding(label_dim, 50),
            nn.Linear(50, 7 * 7),
            nn.ReLU(True)
        )
        self.noise_up_sample = nn.Sequential(
            nn.Linear(noise_dim, 128 * 7 * 7),
            nn.LeakyReLU(True)
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(129, 128, kernel_size=(3, 3), padding=(1, 1)),  # 生成 (128, 7, 7)的图片
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), padding=(1, 1), stride=(2, 2)),  # 生成 (64, 14, 14)的图片
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 1, kernel_size=(4, 4), padding=(1, 1), stride=(2, 2)),  # 生成 (1, 28, 28)的图片
            nn.Tanh()
        )

    def forward(self, X, label):
        label = self.label_up_sample(label)
        label = label.view(-1, 1, 7, 7)
        X = self.noise_up_sample(X)
        X = X.view(-1, 128, 7, 7)
        X = torch.cat([label, X], dim=1)
        return self.net(X)


class Discriminator(nn.Module):
    def __init__(self, class_dim, pic_size, channels):
        super(Discriminator, self).__init__()
        self.channels = channels
        self.pic_size = pic_size
        self.class_dim = class_dim

        self.label_embedding = nn.Sequential(
            nn.Embedding(class_dim, 50),
            nn.Linear(50, pic_size * pic_size * channels),
            nn.ReLU(True)
        )

        self.net = nn.Sequential(
            nn.Conv2d(channels+1, 64, kernel_size=(3, 3), stride=(2, 2)),
            nn.LeakyReLU(),
            # nn.Dropout2d(),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2)),
            nn.LeakyReLU(),
            # nn.Dropout2d(),
            nn.BatchNorm2d(128),
            nn.Flatten(),
            nn.Linear(128*6*6, class_dim+1),
            nn.Sigmoid()
        )

    def forward(self, X, label):
        label = self.label_embedding(label)
        label = label.view(-1, 1, self.pic_size, self.pic_size)
        X = torch.cat([label, X], dim=1)
        return self.net(X)


class CGAN(nn.Module):
    def __init__(self, noise_size, class_dim, pic_size, channels):
        super(CGAN, self).__init__()
        self.generator = Generator(noise_size, class_dim)
        self.discriminator = Discriminator(class_dim, pic_size, channels)

    def forward(self, real_X, noise, labels):
        # 判别真实数据
        real_output = self.discriminator(real_X, labels)
        # 判别生成数据
        fake_output = self.generator(noise, labels)
        d_fake_output = self.discriminator(fake_output.detach(), labels)
        g_fake_output = self.discriminator(fake_output, labels)
        return real_output, d_fake_output, g_fake_output
