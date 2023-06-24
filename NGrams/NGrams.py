from NModule.textProcessing import Vocab, tokenize


class NGrams:
    def __init__(self, corpus, n_grams=2):
        self.n_grams = n_grams
        self.n_grams_seq = self.gen_n_grams(corpus)
        self.vocab = Vocab(self.n_grams_seq)
        self.token_freq = self.vocab.token_freqs

    def gen_n_grams(self, text):
        tokens = []
        for idx in range(len(text) - self.n_grams + 1):
            meta = tuple(text[i] for i in range(idx, idx + self.n_grams))
            tokens.append(meta)
        return tokens



