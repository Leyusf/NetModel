import torch
from PIL import Image
from torch.utils.data import Dataset


def train_gan(net, loss_fn, g_optim, d_optim, epochs, train_loader, batch_size, noise_size, pic_size, device,
              draw_func):
    gen_loss = []
    dis_loss = []
    count = len(train_loader)
    valid = torch.ones((batch_size, 1), requires_grad=False, device=device)
    fake = torch.zeros((batch_size, 1), requires_grad=False, device=device)
    for e in range(1, epochs + 1):
        d_epoch_loss = 0
        g_epoch_loss = 0
        for img, _ in train_loader:
            img = img.to(device)
            noise = torch.randn(batch_size, noise_size, device=device)
            real_output, d_fake_output, g_fake_output = net(img, noise)

            # 优化生成器
            g_optim.zero_grad()
            g_loss = loss_fn(g_fake_output, valid)
            g_loss.backward()
            g_optim.step()

            # 优化判别器
            d_optim.zero_grad()
            real_loss = loss_fn(real_output, valid)
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
            draw_func(net.generator, torch.randn(16, noise_size, device=device))
    return gen_loss, dis_loss


class CMPDataset(Dataset):
    def __init__(self, imgs_path, annos_path, transform):
        self.imgs_path = imgs_path
        self.annos_path = annos_path
        self.transform = transform

    def __getitem__(self, index):
        img, anno = self.imgs_path[index], self.annos_path[index]
        img = Image.open(img)
        img = self.transform(img)
        anno = Image.open(anno)
        anno = anno.convert("RGB")
        anno = self.transform(anno)
        return img, anno

    def __len__(self):
        return len(self.imgs_path)

