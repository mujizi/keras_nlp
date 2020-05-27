# -*- coding: utf-8 -*-
# @Time    : 2020/5/27 下午3:08
# @Author  : Benqi
import pickle
import jieba
import json
import numpy as np
import pandas as pd
from keras import Model
from keras.utils import to_categorical
from tools import read_jsonline, read_json
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Embedding, Dense, Conv1D, Flatten, Input, MaxPool1D, Dropout, LSTM, Bidirectional


def seq_padding(x, pad_len, value=0):
    return np.array([np.concatenate([i, [value] * (pad_len - len(i))]) if len(i) < pad_len else i[:pad_len] for i in x])


def to_onehot(y, label2i):
    length = len(label2i)
    l = []
    for i in y:
        zero = np.array([0] * length)
        zero[int(i[0])] = 1
        l.append(zero)
    return np.array(l)


class Data_generator:
    def __init__(self, data, token_dict, label_dict, batch_size=2):
        self.data = data
        self.batch_size = batch_size
        self.token_dict = token_dict
        self.label_dict = label_dict
        self.max_len = max([len(i['word']) for i in self.data])
        """记录遍历数据集需要几个step"""
        self.steps = len(self.data) // self.batch_size

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = np.arange(len(self.data))
            np.random.shuffle(idxs)
            x_l, y_l = [], []
            for i in idxs:
                """对x，y进行操作"""
                x_l.append(self.data[i]['word'])
                y_l.append(self.data[i]['tag'])
                if self.batch_size == len(x_l) or idxs[-1] == i:
                    """一个batch出队"""
                    x_index = [[self.token_dict.get(word) for word in sen] for sen in x_l]
                    x_padding = seq_padding(x_index, self.max_len)
                    y_index = [[self.label_dict.get(j) for j in i] for i in y_l]
                    y_padding = seq_padding(y_index, self.max_len, self.label_dict["O"])
                    y_one_hot = [to_categorical(i, len(self.label_dict)) for i in y_padding]
                    yield x_padding, y_one_hot
                    x_l, y_l = [], []


class TextCNN:
    def __init__(self, token_dict, generator, v_generator):
        self.token_dict = token_dict
        self.generator = generator
        self.v_generator = v_generator
        self.model = self.build_model()

    def build_model(self):
        input = Input(shape=(140,), dtype='float64')
        embeder = Embedding(len(self.token_dict), 300, input_length=140, trainable=False)
        embed = embeder(input)
        LSTM

        model.summary()
        return model

    def train(self):
        callbacks_list = [
            EarlyStopping(monitor="val_accuracy",
                          patience=1),
            ModelCheckpoint(filepath='TextCNN.h5',
                            monitor='val_loss',
                            save_best_only=True)
        ]

        history = self.model.fit_generator(self.generator.__iter__(),
                                           steps_per_epoch=10000,
                                           epochs=3,
                                           callbacks=callbacks_list,
                                           validation_data=self.v_generator.__iter__(),
                                           nb_val_samples=2000)

        with open('trainHistoryDict.txt', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

    def predict(self):
        pass


if __name__ == '__main__':
    ROOT_PATH = '/Users/ouhon/PycharmProjects/keras_nlp_tutorial/NER/'
    path = ROOT_PATH + 'data/raw_data/dataset.jsonl'
    train_data = read_jsonline(path)
    token_dict = read_json(ROOT_PATH + 'data/raw_data/token2i.json')
    label_dict = read_json(ROOT_PATH + 'data/raw_data/label2i.json')
    generator = Data_generator(train_data, token_dict, label_dict)
    print(generator.max_len)
    print(generator.__iter__().__next__())