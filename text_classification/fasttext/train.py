# -*- coding: utf-8 -*-
# @Time    : 2020/5/19 下午6:51
# @Author  : Benqi
import jieba
import pandas as pd
from constant import ROOT_PATH
from get_token_dict import add_ngram
from tools import write_json, read_json
import keras
from keras import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.layers import Embedding
from keras.preprocessing import sequence
from keras.layers import GlobalAveragePooling1D
import numpy as np


def seq_padding(x, padding):
    return np.array([np.concatenate([i, [0] * (padding - len(i))]) if len(i) < padding else i[:padding] for i in x])


def to_onehot(y, label2i):
    length = len(label2i)
    l = []
    for i in y:
        zero = np.array([0] * length)
        zero[int(i[0])] = 1
        l.append(zero)
    return np.array(l)


class data_generator:
    def __init__(self, data, token_dict, label_dict, batch_size=1):
        self.data = data
        self.batch_size = batch_size
        self.token_dict = token_dict
        self.label_dict = label_dict
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
                x_l.append(self.data[i][0])
                y_l.append(self.data[i][1])
                if self.batch_size == len(x_l) or idxs[-1] == i:
                    """一个batch出队"""
                    x_cut = [jieba.lcut(i) for i in x_l]
                    x_index = [[self.token_dict.get(word) for word in sen] for sen in x_cut]
                    x_index = add_ngram(x_index, self.token_dict, 2)
                    x_padding = seq_padding(x_index, 50)
                    y_index = [[label_dict.get(i)] for i in y_l]
                    y_one_hot = to_onehot(y_index, self.label_dict)
                    yield x_padding, y_one_hot
                    x_l, y_l = [], []


class FastText:
    def __init__(self, token_dict, generator, v_generator):
        self.generator = generator
        self.v_generator = v_generator
        self.token_dict = token_dict
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Embedding(len(self.token_dict), 200, input_length=50))
        model.add(GlobalAveragePooling1D())
        model.add(Dense(14, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer="adam",
                      metrics=['accuracy'])
        model.summary()
        return model

    def train(self):
        callbacks_list = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=1,
            ),
            keras.callbacks.ModelCheckpoint(
                filepath='fastText_model.h5',
                monitor='val_loss',
                save_best_only=True,
            )]

        history = self.model.fit_generator(self.generator.__iter__(), steps_per_epoch=len(generator), epochs=3, callbacks=callbacks_list, validation_data=self.v_generator.__iter__(),
                                 nb_val_samples=2000)

    def predict(self):
        pass


if __name__ == '__main__':
    path = ROOT_PATH + 'data/toutiao/toutiao_train.csv'
    val_path = ROOT_PATH + 'data/toutiao/toutiao_test.csv'
    train_data = pd.read_csv(path)
    val_data = pd.read_csv(val_path)
    token_dict = read_json(ROOT_PATH + 'data/toutiao/token_dict.json')
    label_dict = read_json(ROOT_PATH + 'data/toutiao/label2i.json')
    train_D = [(x, y) for x, y in zip(train_data['x'], train_data['y'])][:10000]
    val_D = [(x, y) for x, y in zip(val_data['x'], val_data['y'])][:5000]
    generator = data_generator(train_D, token_dict, label_dict)
    v_generator = data_generator(val_D, token_dict, label_dict)
    print('build model')
    fast_text_model = FastText(token_dict, generator, v_generator)
    fast_text_model.train()

