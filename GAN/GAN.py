from torch import nn


class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, num_hidden=1, hidden_size=32):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU()
        )
        for _ in range(num_hidden-2):
            self.net.append(nn.Linear(hidden_size, hidden_size))
            self.net.append(nn.ReLU())
        self.net.append(nn.Linear(hidden_size, output_dim))
        self.net.append(nn.Tanh())

    def forward(self, X):
        return self.net(X)


class Discriminator(nn.Module):
    def __init__(self, input_dim, num_hidden=1, hidden_size=32):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LeakyReLU()  # 采用了带泄露的线性整流激活函数，用于引入一定的非线性特征。
        )
        for _ in range(num_hidden - 2):
            self.net.append(nn.Linear(hidden_size, hidden_size))
            self.net.append(nn.LeakyReLU())
        self.net.append(nn.Linear(hidden_size, 1))
        self.net.append(nn.Sigmoid())

    def forward(self, X):
        return self.net(X)


class GAN(nn.Module):
    def __init__(self, input_dim, output_dim, num_hidden=1, hidden_size=32):
        super(GAN, self).__init__()
        self.output_dim = output_dim
        self.generator = Generator(input_dim, self.output_dim**2, num_hidden, hidden_size)
        self.discriminator = Discriminator(self.output_dim**2, num_hidden, hidden_size)

    def forward(self, real_X, noise):
        # 判别真实数据
        real_output = self.discriminator(real_X.view(-1, self.output_dim * self.output_dim))
        # 判别生成数据
        fake_output = self.generator(noise).view(-1, self.output_dim * self.output_dim)
        d_fake_output = self.discriminator(fake_output.detach())
        g_fake_output = self.discriminator(fake_output)
        return real_output, d_fake_output, g_fake_output



