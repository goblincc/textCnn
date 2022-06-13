#-*-coding: UTF-8 -*-
import pandas as pd
import os
import numpy as np


def csv2txt():
    data = pd.read_csv('../data/uid_nick.csv', encoding='utf-8', error_bad_lines=False)
    with open('../data/uid_nick.txt', 'w+', encoding='utf-8') as f:
        for line in data.values:
            if len(line) == 2:
                f.write((str(line[0]).strip() + '\t' + str(line[1]).strip() + '\n'))

def processTxt():
    with open("../sample/pos.txt", 'rt', encoding='utf-8') as f:
        with open('../data/train_sample.txt', 'a+', encoding='utf-8') as f2:
            for line in f:
                f2.write('0' + '\t' + line.strip() + '\n')

def read_stop_words():
    stop_words_list = []
    with open("../data/stop_words.txt", 'rt', encoding='utf-8') as f:
        for line in f:
            stop_words_list.append(line.strip())
    return stop_words_list


def count_words(counter, text):
    for sentence in text:
        for word in sentence:
            counter[word] += 1


def min_distance(s1, s2):
    len1 = len(s1)
    len2 = len(s2)
    dp = np.zeros((len1 + 1, len2 + 1))
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            delta = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j - 1] + delta, min(dp[i - 1][j] + 1, dp[i][j - 1] + 1))
    return dp[len1][len2]


if __name__ == '__main__':
    # csv2txt()
    processTxt()
    print(min_distance("Q、113--066--524", "Q、113、066、524"))

