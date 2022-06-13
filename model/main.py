#-*-coding: UTF-8 -*-
from tensorflow.keras.preprocessing import sequence
from dataer import dataer
from text_cnn import TextCNN
import numpy as np
import config
import tensorflow as tf
from predict import sentence_index
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
import time


def sample_data(x, y):
    pos = []
    neg = []
    y_sample = []
    for i in range(len(x)):
        if y[i] == 1:
            pos.append(x[i])
            y_sample.append(1)
        if y[i] == 0:
            neg.append(x[i])
    indices = np.random.permutation(np.arange(4 * len(pos)))
    neg_sample = []
    for j in indices:
        neg_sample.append(neg[j])
        y_sample.append(0)
    x_sample = np.concatenate([pos, neg_sample])
    print("采样后比例:", np.sum(y_sample)/len(y_sample))
    return x_sample, np.array(y_sample)


def scheduler(epoch):
    if epoch < 3:
        return 0.001
    else:
        lr = 0.001 * tf.math.exp(0.1 * (5 - epoch))
        return lr.numpy()

def cal_model_performance(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    return accuracy, precision, recall, auc


def embedding_weights(model):
    e0 = model.layers[0]
    weights0 = e0.get_weights()[0]
    e1 = model.layers[1]
    weights1 = e1.get_weights()[0]
    e2 = model.layers[2]
    weights2 = e2.get_weights()[0]
    print(weights0.shape)
    print(weights1.shape)
    print(weights2.shape)


if __name__ == '__main__':
    start = time.time()
    x_train, y_train, x_test, y_test = dataer().generate_train_data()

    x_train = sequence.pad_sequences(x_train, maxlen=config.maxlen, padding='post', truncating='post')
    x_test = sequence.pad_sequences(x_test, maxlen=config.maxlen, padding='post', truncating='post')

    x_sample, y_sample = sample_data(x_train, y_train)

    y_test = np.array(y_test)

    np.random.seed(10)
    shuffled_indices = np.random.permutation(np.arange(len(y_sample)))
    x_train_shuffled = x_sample[shuffled_indices]
    y_train_shuffled = y_sample[shuffled_indices]

    model = TextCNN(config.maxlen, config.max_features, config.embedding_dims)

    # exponential_decay = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.005, decay_steps=1000, decay_rate=0.96)

    model.compile(optimizer=tf.keras.optimizers.Adam(0.002),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=['accuracy'])

    reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)

    model.fit(x_train_shuffled, y_train_shuffled, batch_size=config.batch_size, epochs=config.epochs, callbacks=[reduce_lr])

    tf.keras.utils.plot_model(model, to_file='./model.png', show_shapes=True, show_layer_names=True)

    model.summary()
    embedding_weights(model)

    # model.save_weights('./modelfile/my_model.h5')
    model.save('./modelfile')

    score = model.predict(x_test)
    score_pred = np.mat(score.reshape(-1, len(y_test))).tolist()

    with open('../data/pred.txt', "w+", encoding='utf-8') as f:
        for i in score_pred[0]:
            f.write(str(i) + '\n')

    y_pred = [1 if score >= 0.8 else 0 for score in score_pred[0]]
    accuracy, precision, recall, auc = cal_model_performance(y_test, y_pred)
    print("accuracy:", accuracy)
    print("precision:", precision)
    print("recall:", recall)
    print("auc:", auc)
    end = time.time()
    print("耗时：", (end-start))
