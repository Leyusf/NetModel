from NGrams import NGrams
from NModule.textProcessing import tokenize, read_text_file

tokens = tokenize(read_text_file("..//data//text//timemachine.txt"))
corpus = [token for line in tokens for token in line]

ngrams = NGrams(corpus, 2)
print(ngrams.token_freq[:10])
