from torch import nn


class Generator(nn.Module):
    def __init__(self, input_dim):
        super(Generator, self).__init__()
        self.l1 = nn.Linear(input_dim, 256 * 7 * 7)
        self.bn1 = nn.BatchNorm1d(256 * 7 * 7)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=(3, 3), padding=(1, 1)),  # 生成 (128, 7, 7)的图片
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), padding=(1, 1), stride=(2, 2)),  # 生成 (64, 14, 14)的图片
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 1, kernel_size=(4, 4), padding=(1, 1), stride=(2, 2)),  # 生成 (1, 28, 28)的图片
            nn.Tanh()
        )

    def forward(self, X):
        X = self.bn1(self.l1(X))
        X = X.view(-1, 256, 7, 7)
        return self.net(X)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2)),
            nn.LeakyReLU(),
            nn.Dropout2d(),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2)),
            nn.LeakyReLU(),
            nn.Dropout2d(),
            nn.BatchNorm2d(128),
            nn.Flatten(),
            nn.Linear(128*6*6, 1),
            nn.Sigmoid()
        )

    def forward(self, X):
        return self.net(X)


class DCGAN(nn.Module):
    def __init__(self, noise_size):
        super(DCGAN, self).__init__()
        self.generator = Generator(noise_size)
        self.discriminator = Discriminator()

    def forward(self, real_X, noise):
        # 判别真实数据
        real_output = self.discriminator(real_X)
        # 判别生成数据
        fake_output = self.generator(noise)
        d_fake_output = self.discriminator(fake_output.detach())
        g_fake_output = self.discriminator(fake_output)
        return real_output, d_fake_output, g_fake_output
