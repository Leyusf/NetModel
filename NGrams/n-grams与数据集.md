## 贝叶斯模型
语言模型的原理是贝叶斯概率统计。
贝叶斯公式为：

$$
P({x_1}, {x_2}, {x_3},...,{x_T})=\prod_{t=1}^TP({x_t}|{x_1},...,{x_t-1})
$$

例如，一个文本 "Deep learning is fun"出现的概率是:

$$
P(deep,learning,is,fun)=P(deep)P(learning|deep)P(is|learning,deep)P(fun|is,learning,deep)
$$

这里的概率都是由**语料库**([[文本预处理]])的统计得到的。

$$
\hat{P}(learning|deep)={n(learning, deep)\over n(deep)}
$$

其中 $n(x)$ 和 $n(x,x')$ 分别是单个单词和连续单词对的出现**次数**。
但是，对于一些不常见的单词组合，要想找到足够的出现次数来获得准确的估计可能都不容易。 而对于三个或者更多的单词组合，情况会变得更糟。 许多合理的三个单词组合可能是存在的，但是在数据集中却找不到。 除非我们提供某种解决方案，来将这些单词组合指定为非零计数， 否则将无法在语言模型中使用它们。 如果数据集很小，或者单词非常罕见，那么这类单词出现一次的机会可能都找不到。
一种常见的策略是执行某种形式的 *拉普拉斯平滑(Laplace smoothing)*， 具体方法是在所有计数中添加一个小常量。 用 $n$ 表示训练集中的单词总数，用 $m$ 表示唯一单词的数量。
例如：

$$
\hat{P}(X)={n(x)+\epsilon_1/m\over n+\epsilon_1}
$$

$$
\hat{P}(x'|x)={n(x,x')+\epsilon_2 \hat{P}(x')\over n(x)+\epsilon_2}
$$

$$
\hat{P}(x''|x',x)={n(x,x',x'') + \epsilon_3\hat{P}(x'') \over n(x,x'')+\epsilon_e}
$$

其中， $\epsilon_1$ , $\epsilon_2$ 和 $\epsilon_3$ 是超参数。 以 $\epsilon_1$ 为例：当 $\epsilon_1=0$ 时，不应用平滑； 当 $\epsilon_1$ 接近正无穷大时， $\hat{P}(x)$ 接近均匀概率分布 $1/m$ 。

## n-gram

序列的分布上满足一阶马尔可夫性质。阶数越高，依赖关系越长。常用的为一元语法(unigram)、二元语法(bigram)与三元语法(trigram)。
例如:

$$
P(x_1, x_2, x_3)=P(x_1)P(x_2|x_1)P(x_3|x_2)P(x_4|x_3)
$$

生成n-gram词元。
```
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
```
## 数据集

由于序列数据本质上是连续的，因此我们在处理数据时需要解决这个问题。当序列变得太长而不能被模型一次性全部处理时， 我们可能希望拆分这样的序列方便模型读取。

**1.顺序生成小批量数据集**

从语料库中按照顺序截取长度为n的小序列作为数据集返回。
```
def seq_data_iter_sequential(corpus, batch_size, num_steps):  
    """使用顺序分区生成一个小批量子序列"""  
    # 从随机偏移量开始划分序列  
    offset = random.randint(0, num_steps)  
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size  
    Xs = torch.tensor(corpus[offset: offset + num_tokens])  
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])  
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)  
    num_batches = Xs.shape[1] // num_steps  
    for i in range(0, num_steps * num_batches, num_steps):  
        X = Xs[:, i: i + num_steps]  
        Y = Ys[:, i: i + num_steps]  
        yield X, Y
```
**2.随机生成小批量数据集**

从语料库中随机选择长度为n的连续小序列作为数据集返回。

```
def seq_data_iter_random(corpus, batch_size, num_steps):  
    """使用随机抽样生成一个小批量子序列"""  
    # 从随机偏移量开始对序列进行分区，随机范围包括num_steps-1  
    corpus = corpus[random.randint(0, num_steps - 1):]  
    # 减去1，是因为我们需要考虑标签  
    num_subseqs = (len(corpus) - 1) // num_steps  
    # 长度为num_steps的子序列的起始索引  
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))  
    # 在随机抽样的迭代过程中，  
    # 来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻  
    random.shuffle(initial_indices)  
  
    def data(pos):  
        # 返回从pos位置开始的长度为num_steps的序列  
        return corpus[pos: pos + num_steps]  
  
    num_batches = num_subseqs // batch_size  
    for i in range(0, batch_size * num_batches, batch_size):  
        # 在这里，initial_indices包含子序列的随机起始索引  
        initial_indices_per_batch = initial_indices[i: i + batch_size]  
        X = [data(j) for j in initial_indices_per_batch]  
        Y = [data(j + 1) for j in initial_indices_per_batch]  
        yield torch.tensor(X), torch.tensor(Y)
```

**3.数据迭代器**

用于torch的dataloader。
```
class SeqDataLoader:  
    """加载序列数据的迭代器"""  
  
    def __init__(self, batch_size, num_steps, use_random_iter, filename, max_tokens, token="char"):  
        if use_random_iter:  
            self.data_iter_fn = seq_data_iter_random  
        else:  
            self.data_iter_fn = seq_data_iter_sequential  
        self.corpus, self.vocab = load_corpus(filename, max_tokens, token)  
        self.batch_size, self.num_steps = batch_size, num_steps  
  
    def __iter__(self):  
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)
```

**4. 返回数据集的迭代器和词表**
```
def load_data(batch_size, num_steps,  filename, use_random_iter=False, max_tokens=10000, token="char"):  
    """返回数据集的迭代器和词表"""  
    data_iter = SeqDataLoader(  
        batch_size, num_steps, use_random_iter, filename, max_tokens, token)  
    return data_iter, data_iter.vocab
```
