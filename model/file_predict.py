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
import os,sys,time

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
    for word in line:
        if word not in stop_words_list:
            if word in key_dict:
                word = key_dict[word]
            if word in vocab_dict:
                index = vocab_dict[word]
                line_vec.append(index)
    return line_vec


##测试多条样本
def predict_sample(cmd):
    stop_words_list = read_stop_words()
    vocab_dict = load_vocab()
    d = dataer()
    d.keyword_dict()
    vector = []
    cmds = []
    for lines in cmd:
        if lines not in cmds:
            line = lines.replace("\t", "").replace(" ", "")
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
            if len(word_index) >= 3:
                vector.append(word_index)
                cmds.append(lines)
    vectors = sequence.pad_sequences(vector, maxlen=config.maxlen, padding='post', truncating='post')
    scores = model.predict(vectors)
    pred = np.array(scores).reshape(-1, len(vectors))[0]
    with open('../data/pred_file.txt', "a+", encoding='utf-8') as f:
        for k, v in enumerate(pred):
            if v > 0.8:
                print(str(v) + '\t' + cmds[k] + '\n')
                f.write(str(v) + '\t' + cmds[k] + '\n')
            


if __name__ == '__main__':
    print(time.time())
    flag = 'chan'
    path_list = os.listdir("../data/")
    path_list.sort()
    for filename in path_list:
        if filename.startswith(flag):
            cmd = []
            with open("../data/"+filename, 'rt', encoding='utf-8') as f:
                for line in f:
                    conts = line.strip()
                    if len(conts) <= 150:
                        cmd.append(conts)
                        if len(cmd) >= 50000:
                            predict_sample(cmd)
                            cmd = []
            predict_sample(cmd)
    print(time.time())

