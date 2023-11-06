import torch
from torch import nn

from utils import get_all_gpu, read_snli, load_pretrained_model, fine_tune_bert, SNLIBERTDataset


class BERTClassifier(nn.Module):
    def __init__(self, bert):
        super(BERTClassifier, self).__init__()
        self.encoder = bert.encoder
        self.hidden = bert.hidden
        self.output = nn.Linear(256, 3)

    def forward(self, inputs):
        tokens_X, segments_X, valid_lens_x = inputs
        encoded_X = self.encoder(tokens_X, segments_X, valid_lens_x)
        return self.output(self.hidden(encoded_X[:, 0, :]))


devices = get_all_gpu()
bert, vocab = load_pretrained_model(
    '.\\BERT_Model\\bert.small.torch\\', num_hiddens=256, ffn_num_hiddens=512, num_heads=4,
    num_layers=2, dropout=0.1, max_len=512, devices=devices)

print("Reading Data...")
batch_size, max_len, num_workers = 512, 128, 8
train_set = SNLIBERTDataset(read_snli('..\\data\\snli_1.0\\', True), max_len, vocab, num_workers)
test_set = SNLIBERTDataset(read_snli('..\\data\\snli_1.0\\', False), max_len, vocab, num_workers)

train_iter = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True)
test_iter = torch.utils.data.DataLoader(test_set, batch_size)
print("Finish reading")

net = BERTClassifier(bert)
lr, num_epochs = 1e-4, 5
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction='none')
print("Training...")
fine_tune_bert(net, train_iter, test_iter, loss, trainer, num_epochs)
print("Finish")
