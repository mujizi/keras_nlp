# -*- coding: utf-8 -*-
# @Time    : 2020/5/21 下午2:35
# @Author  : Benqi

import pickle
import jieba
import json
import codecs
import pandas as pd
import numpy as np
from constant import dict_path, config_path, checkpoint_path
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras.layers import Input, Dense, Lambda
from keras import Model
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
    def __init__(self, data, dict_path, label_dict, batch_size=1):
        self.data = data
        self.batch_size = batch_size
        self.dict_path = dict_path
        self.label_dict = label_dict
        self.tokenizer = self.get_token_dict()

    def get_token_dict(self):
        self.token_dict = {}
        with codecs.open(self.dict_path, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                self.token_dict[token] = len(self.token_dict)
        tokenizer = Tokenizer(self.token_dict)
        return tokenizer

    def __iter__(self):
        while True:
            idxs = np.arange(len(self.data))
            np.random.shuffle(idxs)
            x_l1, x_l2, y_l = [], [], []
            for i in idxs:
                text = self.data[i][0]
                x1, x2 = self.tokenizer.encode(first=text)
                x_l1.append(x1)
                x_l2.append(x2)
                y_l.append(self.data[i][1])
                if self.batch_size == len(x_l1) or idxs[-1] == i:
                    X1 = seq_padding(x_l1, 50)
                    X2 = seq_padding(x_l2, 50)
                    y_index = [[label_dict.get(i)] for i in y_l]
                    y_one_hot = to_onehot(y_index, self.label_dict)
                    yield [X1, X2], y_one_hot
                    x_l1, x_l2, y_l = [], [], []


class BertCls:
    def __init__(self, dict_path, config_path, checkpoint_path, generator, v_generator):
        self.dict_path = dict_path
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.generator = generator
        self.v_generator = v_generator
        self.model = self.build_model()

    def build_model(self):
        bert_model = load_trained_model_from_checkpoint(self.config_path, self.checkpoint_path, seq_len=None)

        for l in bert_model.layers:
            l.trainable = True

        input1 = Input(shape=(None,))
        input2 = Input(shape=(None,))
        x = bert_model([input1, input2])
        x = Lambda(lambda x: x[:, 0])(x)
        output = Dense(14, use_bias=False, activation='softmax')(x)
        bert_cls = Model([input1, input2], output)
        bert_cls.compile(loss='categorical_crossentropy',
                         optimizer='adam',
                         metrics=['accuracy'])
        bert_cls.summary()
        return bert_cls

    def train(self):
        callbacks_list = [
            EarlyStopping(monitor='val_accuracy',
                          patience=1),
            ModelCheckpoint(filepath='bert_cls.h5',
                            monitor='val_loss',
                            save_best_only=True)
        ]

        self.model.fit_generator(generator.__iter__(),
                                 steps_per_epoch=10000,
                                 epochs=3,
                                 callbacks=callbacks_list,
                                 validation_data=v_generator.__iter__(),
                                 nb_val_samples=2000)

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
    generator = Data_generator(train_D, dict_path, label_dict)
    v_generator = Data_generator(val_D, dict_path, label_dict)
    # bert_cls = BertCls(dict_path, config_path, checkpoint_path, generator, v_generator)
    # bert_cls.train()
    [x1, x2], y = generator.__iter__().__next__()
    print([x1,x2])
    print(y)


