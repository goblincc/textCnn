from vocab import Vocab
from collections import Counter
from utils import count_words
from utils import read_stop_words
import numpy as np
import config

class dataer(object):

    def __init__(self):
        self.key_dict = {}
        self.x_train = []
        self.train_txt_list = []
        self.test_txt_list = []
        self.max_vocab_size = 1000000
        self.vocab = Vocab()

    def keyword_dict(self):
        with open("../data/tb_ks_keywd_reflect_keyword.txt", "rt", encoding='utf-8') as f:
            for line in f:
                s = line.split('\t')
                if s[0].strip() not in self.key_dict:
                    self.key_dict[s[0].strip()] = s[1].strip()
        with open("../data/tb_ks_keywd_reflect_lettnum.txt", "rt", encoding='utf-8') as f:
            for line in f:
                s = line.split('\t')
                if s[0].strip() not in self.key_dict:
                    self.key_dict[s[0].strip()] = s[1].strip()

    def replace_keyword(self):
        y_train_data = self.read_data_replace(config.train_path, 'train')
        y_test_data = self.read_data_replace(config.test_path, 'test')
        return y_train_data, y_test_data

    def read_data_replace(self, path, flag):
        stop_words_list = read_stop_words()
        y = []
        with open(path, 'rt', encoding='utf-8') as f:
            for line in f:
                s = line.strip().split('\t')
                if len(s) == 2:
                    if s[0].isdigit() and len(s[0]) == 1:
                        txt = []
                        for item in s[1]:
                            if item not in stop_words_list:
                                if item in self.key_dict:
                                    char = self.key_dict[item]
                                    txt.append(char)
                                else:
                                    txt.append(item)
                        # 替换完停用词后长度超过4的纳入训练数据
                        if len(txt) >= 3:
                            y.append(int(s[0].strip()))
                            if flag == 'train':
                                self.train_txt_list.append(txt)
                            elif flag == 'test':
                                self.test_txt_list.append(txt)
        return y

    def build_vocab(self):
        word_counts = Counter()
        count_words(word_counts, self.train_txt_list)
        for words, count in word_counts.most_common(self.max_vocab_size):
            self.vocab.add_words([words])
        with open('../data/vocab_index.txt', "w+", encoding='utf-8') as f:
            for keys, values in self.vocab.word2index.items():
                f.write(keys + '\t' + str(values) + '\n')

    def sentence_vec(self):
        x_train_vec = []
        x_test_vec = []
        for sentence in self.train_txt_list:
            index_sen = []
            for word in sentence:
                if word not in self.vocab.word2index:
                    continue
                index_sen.append(self.vocab[word])
            x_train_vec.append(index_sen)
        for sentence in self.test_txt_list:
            index_sen = []
            for word in sentence:
                if word not in self.vocab.word2index:
                    continue
                index_sen.append(self.vocab[word])
            x_test_vec.append(index_sen)
        return x_train_vec, x_test_vec

    def generate_train_data(self):
        self.keyword_dict()
        y_train, y_test = self.replace_keyword()
        print("违规样本占比：", np.sum(y_train)/len(y_train))
        self.build_vocab()
        x_train, x_test = self.sentence_vec()
        return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    d = dataer()
    d.keyword_dict()
    x_train, y_train, x_test, y_test = d.generate_train_data()





