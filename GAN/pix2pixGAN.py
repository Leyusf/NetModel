import torch
from torch import nn
import torch.nn.functional as F


class UpSampler(nn.Module):
    #  上采样器
    def __init__(self, in_channels, out_channels):
        super(UpSampler, self).__init__()
        self.upconv_relu = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                               output_padding=(1, 1)),
            nn.LeakyReLU(inplace=True),
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, X, is_drop=False):
        X = self.upconv_relu(X)
        X = self.bn(X)
        if is_drop:
            X = F.dropout(X)
        return X


class DownSampler(nn.Module):
    #  下采样器
    def __init__(self, in_channels, out_channels):
        super(DownSampler, self).__init__()
        self.downconv_relu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(inplace=True)
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, X, is_bn=True):
        X = self.downconv_relu(X)
        if is_bn:
            X = self.bn(X)
        return X


class Generator(nn.Module):
    def __init__(self, out_channels):
        super(Generator, self).__init__()
        self.down1 = DownSampler(3, 64)  # 3是图片的通道数
        self.down2 = DownSampler(64, 128)
        self.down3 = DownSampler(128, 256)
        self.down4 = DownSampler(256, 512)
        self.down5 = DownSampler(512, 512)
        self.down6 = DownSampler(512, 512)
        self.up1 = UpSampler(512, 512)
        self.up2 = UpSampler(1024, 512)
        self.up3 = UpSampler(1024, 256)
        self.up4 = UpSampler(512, 128)
        self.up5 = UpSampler(256, out_channels)
        self.last = nn.ConvTranspose2d(2*out_channels, 3, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                                       output_padding=(1, 1))

    def forward(self, X):
        x1 = self.down1(X)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)

        x6 = self.up1(x6, True)
        x6 = torch.cat([x6, x5], dim=1)

        x6 = self.up2(x6, True)
        x6 = torch.cat([x6, x4], dim=1)

        x6 = self.up3(x6, True)
        x6 = torch.cat([x6, x3], dim=1)

        x6 = self.up4(x6, True)
        x6 = torch.cat([x6, x2], dim=1)

        x6 = self.up5(x6, True)
        x6 = torch.cat([x6, x1], dim=1)

        x6 = torch.tanh(self.last(x6))
        return x6


class Discriminator(nn.Module):
    def __init__(self, out_channels):
        super(Discriminator, self).__init__()
        self.down1 = DownSampler(6, 64)  # 6是anno和img拼接的通道数
        self.down2 = DownSampler(64, 128)  # 70 * 70 的 patch
        self.conv1 = nn.Conv2d(128, out_channels, kernel_size=(3, 3))
        self.bn = nn.BatchNorm2d(out_channels)
        self.last = nn.Conv2d(out_channels, 1, kernel_size=(3, 3))

    def forward(self, img, anno):
        X = torch.cat([anno, img], dim=1)
        X = self.down1(X, False)
        X = self.down2(X)
        X = self.conv1(X)
        X = F.leaky_relu(X)
        X = F.dropout(self.bn(X))
        X = torch.sigmoid(self.last(X))  # 1*60*60
        return X


class Pix2PixGAN(nn.Module):
    def __init__(self, gen_channels, dis_channel):
        super(Pix2PixGAN, self).__init__()
        self.generator = Generator(gen_channels)
        self.discriminator = Discriminator(dis_channel)

    def forward(self, img, anno):
        # 判别真实数据
        real_output = self.discriminator(img, anno)
        # 判别生成数据
        fake_output = self.generator(anno)
        d_fake_output = self.discriminator(fake_output.detach(), anno)
        g_fake_output = self.discriminator(fake_output, anno)
        return real_output, d_fake_output, g_fake_output, fake_output
