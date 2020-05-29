# -*- coding: utf-8 -*-
# @Time    : 2020/5/27 下午3:08
# @Author  : Benqi

import pickle
import numpy as np
from keras import Model
from bert4keras.layers import ConditionalRandomField
from tools import read_jsonline, read_json
from keras.layers import Embedding, Dense, Input, LSTM, Bidirectional, TimeDistributed
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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
    def __init__(self, data, token_dict, label_dict, batch_size=10):
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
                    x_padding = seq_padding(x_index, 140)
                    y_index = [[self.label_dict.get(j) for j in i] for i in y_l]
                    y_padding = seq_padding(y_index, 140, self.label_dict["O"])
                    yield x_padding, y_padding
                    x_l, y_l = [], []


class LstmCrf:
    def __init__(self, token_dict, generator, v_generator):
        self.token_dict = token_dict
        self.generator = generator
        self.v_generator = v_generator
        self.model = self.build_model()

    def build_model(self):
        input = Input(shape=(140,), dtype='float64')
        embeder = Embedding(len(self.token_dict), 300, input_length=140, trainable=False)
        embed = embeder(input)
        bilstm = Bidirectional(LSTM(units=300,
                             return_sequences=True,
                             dropout=0.5,
                             recurrent_dropout=0.5))(embed)
        lstm = LSTM(units=300 * 2,
                    return_sequences=True,
                    dropout=0.5,
                    recurrent_dropout=0.5)(bilstm)
        out = TimeDistributed(Dense(17, activation='relu'))(lstm)
        CRF = ConditionalRandomField(17)
        out = CRF(out)
        model = Model(input, out)
        model.compile(optimizer='adam', loss=CRF.sparse_loss, metrics=[CRF.sparse_accuracy])
        model.summary()
        return model

    def train(self):
        history = self.model.fit_generator(self.generator.__iter__(),
                                           steps_per_epoch=len(self.generator),
                                           epochs=10)

        with open('trainHistoryDict.txt', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

    def predict(self):
        pass


if __name__ == '__main__':
    ROOT_PATH = '/Users/ouhon/PycharmProjects/keras_nlp_tutorial/NER/lstm_crf/'
    path = ROOT_PATH + 'data/raw_data/dataset.jsonl'
    model_path = ROOT_PATH + 'lstm_crf.h5'
    data = read_jsonline(path)
    train_data = data[:int(len(data) * 0.8)]
    val_data = data[int(len(data) * 0.8):]
    token_dict = read_json(ROOT_PATH + 'data/raw_data/token2i.json')
    label_dict = read_json(ROOT_PATH + 'data/raw_data/label2i.json')
    i2label = {str(v): str(i) for i, v in label_dict.items()}
    generator = Data_generator(train_data, token_dict, label_dict)
    v_generator = Data_generator(val_data, token_dict, label_dict)
    lstm_crf = LstmCrf(token_dict, generator, v_generator)
    lstm_crf.train()
