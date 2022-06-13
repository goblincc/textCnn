from collections import Counter
import numpy as np

class Vocab(object):
    def __init__(self):
        self.word2index = {}
        self.word2count = Counter()
        self.index2word = []

    def add_words(self, words):
        for word in words:
            if word not in self.word2index:
                self.word2index[word] = len(self.index2word)
                self.index2word.append(word)
        self.word2count.update(words)

    def __getitem__(self, item):
        if type(item) is int:
            return self.index2word[item]
        return self.word2index.get(item)

    def __len__(self):
        return len(self.index2word)

    def size(self):
        return len(self.index2word)
