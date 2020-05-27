# -*- coding: utf-8 -*-
# @Time    : 2020/5/21 上午11:41
# @Author  : Benqi
import pickle
import jieba
import json
import pandas as pd
import numpy as np
from keras import Model
from keras.layers import Embedding, Dense, Conv1D, Flatten, Input, MaxPool1D, Dropout
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint


def read_json(src_filename):
    """
    read json file
    :param src_filename: source file path
    :return: loaded object
    """
    with open(src_filename, encoding='utf-8') as f:
        return json.load(f)


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


class Data_generator:
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
                    x_padding = seq_padding(x_index, 50)
                    y_index = [[self.label_dict.get(i)] for i in y_l]
                    y_one_hot = to_onehot(y_index, self.label_dict)
                    yield x_padding, y_one_hot
                    x_l, y_l = [], []


class TextCNN:
    def __init__(self, token_dict, generator, v_generator):
        self.token_dict = token_dict
        self.generator = generator
        self.v_generator = v_generator
        self.model = self.build_model()

    def build_model(self):
        input = Input(shape=(50,), dtype='float64')
        embeder = Embedding(len(self.token_dict), 300, input_length=50, trainable=False)
        embed = embeder(input)
        conv3 = Conv1D(256, 3, padding='valid', strides=1, activation='relu')(embed)
        conv3 = MaxPool1D(pool_size=48)(conv3)
        conv4 = Conv1D(256, 4, padding='valid', strides=1, activation='relu')(embed)
        conv4 = MaxPool1D(pool_size=47)(conv4)
        conv5 = Conv1D(256, 5, padding='valid', strides=1, activation='relu')(embed)
        conv5 = MaxPool1D(pool_size=46)(conv5)
        cnn = concatenate([conv3, conv4, conv5], axis=-1)
        flat = Flatten()(cnn)
        drop = Dropout(0.2)(flat)
        output = Dense(14, activation='softmax')(drop)
        model = Model(inputs=input, outputs=output)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

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
    ROOT_PATH = '/Users/ouhon/PycharmProjects/keras_nlp_tutorial/text_classification/'
    path = ROOT_PATH + 'data/toutiao/toutiao_train.csv'
    val_path = ROOT_PATH + 'data/toutiao/toutiao_test.csv'
    train_data = pd.read_csv(path)
    val_data = pd.read_csv(val_path)
    token_dict = read_json(ROOT_PATH + 'data/toutiao/token_dict.json')
    label_dict = read_json(ROOT_PATH + 'data/toutiao/label2i.json')
    train_D = [(x, y) for x, y in zip(train_data['x'], train_data['y'])][:10000]
    val_D = [(x, y) for x, y in zip(val_data['x'], val_data['y'])][:5000]
    generator = Data_generator(train_D, token_dict, label_dict)
    v_generator = Data_generator(val_D, token_dict, label_dict)
    text_cnn = TextCNN(token_dict, generator, v_generator)
    text_cnn.train()
