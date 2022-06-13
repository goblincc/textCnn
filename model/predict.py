#-*-coding: UTF-8 -*-

from tensorflow.keras import Model
from tensorflow.keras import models
from tensorflow.keras.preprocessing import sequence
import config
from text_cnn import TextCNN
from dataer import dataer
import numpy as np
from utils import read_stop_words
from datetime import datetime
import time

model = models.load_model('./modelfile')

def load_vocab():
    vocab_dict = {}
    with open("../data/vocab_index.txt", 'rt', encoding='utf-8') as f:
        for line in f:
            s = line.split('\t')
            vocab_dict[s[0].strip()] = s[1].strip()
    return vocab_dict


def sentence_index(line):
    stop_words_list = read_stop_words()
    vocab_dict = load_vocab()
    d = dataer()
    d.keyword_dict()
    key_dict = d.key_dict
    line_vec = []
    words = []
    for word in line:
        if word not in stop_words_list:
            if word in key_dict:
                word = key_dict[word]
            if word in vocab_dict:
                index = vocab_dict[word]
                line_vec.append(index)
            words.append(word)
    print(''.join(words))
    return line_vec


##测试多条样本
def predict_sample():
    stop_words_list = read_stop_words()
    vocab_dict = load_vocab()
    d = dataer()
    d.keyword_dict()
    vector = []
    with open("../data/neg2.txt", 'rt', encoding='utf-8') as f:
        for line in f:
            word_index = []
            for word in line:
                if word not in stop_words_list:
                    if word in d.key_dict:
                        word = d.key_dict[word]
                    if word in vocab_dict:
                        word = vocab_dict[word]
                    else:
                        continue
                    word_index.append(word)
            if len(vector) >= 3:
                vector.append(word_index)
    vectors = sequence.pad_sequences(vector, maxlen=config.maxlen, padding='post', truncating='post')
    scores = model.predict(vectors)
    pred = np.array(scores).reshape(-1, len(vectors))[0]
    rate = np.sum([1 if i >= 0.8 else 0 for i in pred])/len(pred)
    with open('../data/pred_neg2.txt', "w+", encoding='utf-8') as f:
        for i in pred:
            f.write(str(i) + '\n')
    print("rate:", rate)


if __name__ == '__main__':
    start = time.time()
    test_content = "架架架"
    line_vector = sentence_index(test_content)
    if len(line_vector) >= 3:
        print(line_vector)
        line_vector = sequence.pad_sequences([line_vector], maxlen=config.maxlen, padding='post', truncating='post')
        score = model.predict(line_vector)
        print(score)
        end = time.time()
        print("time:", (end - start))
    else:
        print(line_vector)
        print("文本长度小于4，请重新输入")

    # predict_sample()
