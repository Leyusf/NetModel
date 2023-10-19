import collections
import math
import os
import shutil
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

from NModule.machineTranslating import truncate_pad
from NModule.utils import Accumulator, Timer
from Seq2Seq import MaskedSoftmaxCELoss


def read_csv_labels(file):
    """
    :param file: csv文件地址
    :return: 文件名 + 标签
    """
    with open(file, "r") as f:
        lines = f.readlines()[1:]
        tokens = [line.rstrip().split(',') for line in lines]
        return dict(((name, label) for name, label in tokens))


def copy_to_dir(filename, target_dir):
    """将对应的图片放到文件夹中"""
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)


def reorg_train_valid(data_dir, labels, valid_ratio):
    """将验证集从原始的训练集中拆分出来"""
    # 训练数据集中样本最少的类别中的样本数
    n = collections.Counter(labels.values()).most_common()[-1][1]
    # 验证集中每个类别的样本数
    n_valid_per_label = max(1, math.floor(n * valid_ratio))
    label_count = {}
    for train_file in os.listdir(os.path.join(data_dir, 'train')):
        label = labels[train_file.split('.')[0]]
        fname = os.path.join(data_dir, 'train', train_file)
        copy_to_dir(fname, os.path.join(data_dir, 'train_valid_test',
                                        'train_valid', label))
        if label not in label_count or label_count[label] < n_valid_per_label:
            copy_to_dir(fname, os.path.join(data_dir, 'train_valid_test',
                                            'valid', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            copy_to_dir(fname, os.path.join(data_dir, 'train_valid_test',
                                            'train', label))
    return n_valid_per_label


def reorg_test(data_dir):
    """在预测期间整理测试集，以方便读取"""
    for test_file in os.listdir(os.path.join(data_dir, 'test')):
        copy_to_dir(os.path.join(data_dir, 'test', test_file),
                    os.path.join(data_dir, 'train_valid_test', 'test',
                                 'unknown'))


def reorg_data(data_dir, valid_ratio):
    labels = read_csv_labels(os.path.join(data_dir, 'labels.csv'))
    reorg_train_valid(data_dir, labels, valid_ratio)
    reorg_test(data_dir)


def accuracy(y_hat, y):
    y_hat = torch.argmax(y_hat, dim=1)
    return (y_hat == y.reshape(y_hat.shape, -1)).sum().float() / y.numel()


def evaluate_model(net, loss_fn, device, dataloader):
    """
    :param net: 模型
    :param loss_fn: 损失函数
    :param device: 设备
    :param dataloader: 使用dataloader来降低内存
    :return: 正确率， 损失
    """
    correct = 0.0
    loss = 0.0
    num = 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            loss += loss_fn(y_hat, y).sum()
            y_hat = torch.argmax(y_hat, dim=1)
            correct += (y_hat == y.reshape(y_hat.shape, -1)).sum().float()
            num += y.numel()
    return correct / num, loss / num


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


def train_model(net, loss_fn, optimizer, epochs, device, dataloader, test_dataloader, save_best=False, init=None,
                scheduler=None):
    if init is False:
        pass
    elif not init:
        net.apply(init_weights)
    else:
        net.apply(init)
    """
    没有早停设计
    :param net: 模型
    :param loss_fn: 损失函数
    :param optimizer: 优化器
    :param epochs: 训练轮数
    :param device: 设备
    :param dataloader: 训练数据的dataloader
    :param test_dataloader: 验证数据的dataloader
    :param save_best: 是否保存最优的模型
    :param init: 权重初始化方法
    :return: 训练精度，损失，验证精度，损失
    """
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
    net.to(device)
    timer = Timer()
    best_acc = 0
    print("Training starting")
    for ep in range(1, epochs + 1):
        correct = 0.0
        loss = 0.0
        num = 0
        net.train()
        timer.start()
        for X, y in dataloader:
            optimizer.zero_grad()
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            loss_ep = loss_fn(y_hat, y).sum()
            loss_ep.backward()
            optimizer.step()
            with torch.no_grad():
                y_hat = torch.argmax(y_hat, dim=1)
                correct += (y_hat == y.reshape(y_hat.shape, -1)).sum().float()
                num += y.numel()
                loss += loss_ep
        timer.stop()

        with torch.no_grad():
            acc = correct / num
            loss = loss / num

        if scheduler:
            scheduler.step()
        net.eval()
        test_acc_ep, test_ls = evaluate_model(net, loss_fn, device, test_dataloader)

        train_loss.append(loss.cpu())
        test_loss.append(test_ls.cpu())

        train_acc.append(acc.cpu())
        test_acc.append(test_acc_ep.cpu())

        if save_best and best_acc < test_acc[-1]:
            best_acc = test_acc[-1]
            torch.save(net.state_dict(), "Best_Param.pt")

        print(f'Epoch: {ep}, train loss: {loss:.3f}, test loss: {test_ls:.3f}, train acc: {acc:.3f}, '
              f'test acc: {test_acc_ep:.3f}')
        print(f'{num / timer.times[-1]:.1f} examples/sec '
              f'on {str(device)} total training time:{timer.sum():.1f} sec')

    return train_loss, test_loss, train_acc, test_acc


def drawGraph(train_ls, test_ls, train_acc, test_acc):
    """
    绘制验证集和训练集的损失与精度变换
    :param train_ls: 训练损失
    :param test_ls: 验证损失
    :param train_acc: 训练精度
    :param test_acc: 验证精度
    :return:
    """
    epochs = [i for i in range(1, 1 + len(train_ls))]
    plt.plot(epochs, train_ls, label='train loss')
    plt.plot(epochs, test_ls, label='test loss')
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

    plt.plot(epochs, train_acc, label='train accuracy')
    plt.plot(epochs, test_acc, label='test accuracy')
    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()


def train_object_detection_model(net, loss_fn, optimizer, epochs, device, dataloader, test_dataloader, save_best=False,
                                 init=None, scheduler=None):
    if init is False:
        pass
    elif not init:
        net.apply(init_weights)
    else:
        net.apply(init)
    """
    没有早停设计
    :param net: 模型
    :param loss_fn: 损失函数
    :param optimizer: 优化器
    :param epochs: 训练轮数
    :param device: 设备
    :param dataloader: 训练数据的dataloader
    :param test_dataloader: 验证数据的dataloader
    :param save_best: 是否保存最优的模型
    :param init: 权重初始化方法
    :return: 训练精度，损失，验证精度，损失
    """
    train_loss = []
    test_loss = []
    net.to(device)
    timer = Timer()
    best_loss = np.inf
    print("Training starting")
    for ep in range(1, epochs + 1):
        loss = []
        num = []
        net.train()
        timer.start()
        for X, y in dataloader:
            optimizer.zero_grad()
            X = X.to(device)
            y = y.to(device)
            output = net(X)
            print(loss_fn[0](output[:, 1], y[:, 0]))
            print(loss_fn[1](y[1:], output[:, 1:]))
            loss_ep = loss_fn[0](output[:, 1], y[:, 0]) + loss_fn[1](y[1:], output[:, 1:])
            print(loss_ep)
            loss_ep.backward()
            optimizer.step()
            with torch.no_grad():
                loss.append(loss_ep)
                num.append(y.numel())
        timer.stop()

        with torch.no_grad():
            num = torch.Tensor(num)
            loss = (torch.Tensor(loss) @ num)
            num = num.sum()
            loss = loss / num

        if scheduler:
            scheduler.step()
        net.eval()
        test_ls = evaluate_object_detection_model(net, loss_fn, device, test_dataloader)

        train_loss.append(loss)
        test_loss.append(test_ls)

        if save_best and best_loss > test_loss[-1]:
            best_loss = test_loss[-1]
            torch.save(net.state_dict(), "Best_Param.pt")

        print(f'Epoch: {ep}, train loss: {loss:.3f}, test loss: {test_ls:.3f}')
        print(f'{num / timer.times[-1]:.1f} examples/sec '
              f'on {str(device)} total training time:{timer.sum():.1f} sec')

    return train_loss, test_loss


def evaluate_object_detection_model(net, loss_fn, device, dataloader):
    """
    :param net: 模型
    :param loss_fn: 损失函数
    :param device: 设备
    :param dataloader: 使用dataloader来降低内存
    :return: 正确率， 损失
    """
    loss = []
    num = []
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            label, bbox = net(X)
            loss.append(loss_fn[0](label, y[0]) + loss_fn[1](y[1:], bbox))
            num.append(y.numel())

        num = torch.Tensor(num)
        loss = (torch.Tensor(loss) @ num) / num.sum()
    return loss


def train_seq(net, loss_fn, optimizer, epochs, train_loader, test_loader, device):
    net.apply(init_weights)
    net.to(device)
    for epoch in range(epochs):
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            l = loss_fn(net(X), y)
            l.sum().backward()
            optimizer.step()
        print(f'epoch {epoch + 1}, train loss: {evaluate_model(net, loss_fn, device, train_loader)[1]:f},'
              f' test loss: {evaluate_model(net, loss_fn, device, test_loader)[1]:f}')


def predict_rnn(prefix, num_preds, net, vocab, device):
    """在prefix后面生成新字符"""
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    for y in prefix[1:]:  # 预热期
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # 预测num_preds步
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])


def grad_clipping(net, theta):  # @save
    """裁剪梯度"""
    params = [p for p in net.parameters() if p.requires_grad]
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def train_epoch_rnn(net, train_iter, loss, updater, device, use_random_iter, theta=1):
    state, timer = None, Timer()
    metric = Accumulator(2)  # 训练损失之和,词元数量
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 在第一次迭代或使用随机抽样时初始化state
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # state对于nn.GRU是个张量
                state.detach_()
            else:
                # state对于nn.LSTM是个张量
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)

        l = loss(y_hat, y.long())

        updater.zero_grad()
        l.backward()
        grad_clipping(net, theta)
        updater.step()

        metric.add(l * y.numel(), y.numel())
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()


def train_rnn(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False, theta=1):
    loss = nn.CrossEntropyLoss()

    updater = torch.optim.SGD(net.parameters(), lr)
    predict = lambda prefix: predict_rnn(prefix, 50, net, vocab, device)
    # 训练和预测
    graph = []
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_rnn(
            net, train_iter, loss, updater, device, use_random_iter, theta)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            graph.append((epoch + 1, ppl))
    drawPerplexity(graph)
    print(f'Perplexity {ppl:.1f}, {speed:.1f} token/sec on {str(device)}')


def xavier_init_weight_seq2seq(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    elif type(m) == nn.GRU:
        for name, param in m.named_parameters():
            if name == "weight":
                nn.init.xavier_uniform_(param)


def train_seq2seq(net, train_iter, vocab, lr, num_epochs, device, theta=1):
    net.apply(xavier_init_weight_seq2seq)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    graph = []
    print("Training ...")
    for epoch in range(num_epochs):
        timer = Timer()
        metric = Accumulator(2)
        for batch in train_iter:
            optimizer.zero_grad()
            x, x_valid_len, y, y_valid_len = [x.to(device) for x in batch]
            # 添加一个句子开始标志
            bos = torch.tensor([vocab['<bos>']] * y.shape[0], device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, y[:, :-1]], dim=1)  # 强制教学
            y_hat, _ = net(x, dec_input, x_valid_len)
            l = loss(y_hat, y, y_valid_len)
            l.sum().backward()
            grad_clipping(net, theta)
            optimizer.step()
            num_tokens = y_valid_len.sum()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            graph.append((epoch + 1, metric[0] / metric[1]))
        print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
              f'tokens/sec on {str(device)}')
    drawPerplexity(graph, "loss", "loss-epoch", ylabel="loss")


def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps, device, save_attention_weights=False):
    net.eval()
    src_token = src_vocab[src_sentence.lower().split(" ")] + [src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_token)], device=device)
    src_token = truncate_pad(src_token, num_steps, src_vocab['<pad>'])
    # 添加批量轴
    enc_X = torch.unsqueeze(torch.tensor(src_token, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # 添加批量轴
    dec_X = torch.unsqueeze(torch.tensor([tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # 我们使用具有预测最高可能性的词元，作为解码器在下一时间步的输入
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # 保存注意力权重
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # 一旦序列结束词元被预测，输出序列的生成就完成了
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq


def bleu(pred_seq, label_seq, k):
    """计算BLEU"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score


def drawPerplexity(data, label='perplexity', title="perplexity-Epoch", xlabel="epoch", ylabel="Perplexity"):
    epochs = [d[0] for d in data]
    ppl = [d[1] for d in data]
    plt.plot(epochs, ppl, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()
