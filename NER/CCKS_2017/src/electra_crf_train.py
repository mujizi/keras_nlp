# -*- coding: utf-8 -*-
# @Time    : 2020/5/29 下午2:36
# @Author  : Benqi

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import random
import codecs
import numpy as np
from tqdm import tqdm
from tools import read_jsonline
from evalution import get_ner_fmeasure
from constant import dict_path, config_path, checkpoint_path
from bert4keras.snippets import ViterbiDecoder
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import open, ViterbiDecoder
from bert4keras.layers import ConditionalRandomField
from keras.layers import Dense
from keras.models import Model


def seq_padding(x, pad_len, value=0):
    return np.array([np.concatenate([i, [value] * (pad_len - len(i))]) if len(i) < pad_len else i[:pad_len] for i in x])


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]')  # 剩余的字符是[UNK]
        return R


class Data_generator:
    def __init__(self, data, dict_path, label_dict, max_len, batch_size=40):
        self.data = data
        self.batch_size = batch_size
        self.dict_path = dict_path
        self.label_dict = label_dict
        self.max_len = max_len
        self.tokenizer = self.get_token_dict()
        self.steps = len(self.data) // batch_size

    def __len__(self):
        return self.steps

    def get_token_dict(self):
        self.token_dict = {}
        with codecs.open(self.dict_path, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                self.token_dict[token] = len(self.token_dict)
        tokenizer = OurTokenizer(self.token_dict)
        return tokenizer

    def __iter__(self):
        while True:
            idxs = np.arange(len(self.data))
            np.random.shuffle(idxs)
            x_l1, x_l2, y_l = [], [], []
            for i in idxs:
                text = self.data[i]['content']
                text = ''.join(text)
                text = text[:510]
                tag = self.data[i]['tag'][:510]
                tag = ['O'] + tag + ['O']
                x1, x2 = self.tokenizer.encode(text)
                x_l1.append(x1)
                x_l2.append(x2)
                y_l.append(tag)
                if self.batch_size == len(x_l1) or idxs[-1] == i:
                    X1 = seq_padding(x_l1, self.max_len)
                    X2 = seq_padding(x_l2, self.max_len)
                    y_index = [[self.label_dict.get(j) for j in i] for i in y_l]
                    y_padding = seq_padding(y_index, self.max_len, self.label_dict['O'])
                    yield [X1, X2], y_padding
                    x_l1, x_l2, y_l = [], [], []


class PretrainCrf:
    def __init__(self, dict_path, config_path, checkpoint_path, generator):
        self.dict_path = dict_path
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.generator = generator
        self.model = self.build_model()

    def build_model(self):
        model = build_transformer_model(
            self.config_path,
            self.checkpoint_path,
            model='electra'
        )
        output_layer = 'Transformer-%s-FeedForward-Norm' % (12 - 1)
        output = model.get_layer(output_layer).output
        output = Dense(11)(output)
        self.CRF = ConditionalRandomField(lr_multiplier=100)
        output = self.CRF(output)

        model = Model(model.input, output)
        model.summary()
        model.compile(loss=self.CRF.sparse_loss,
                      optimizer=Adam(1e-4),
                      metrics=[self.CRF.sparse_accuracy]
                      )
        return model

    def train(self, eva):
        self.model.fit_generator(self.generator.__iter__(),
                                 steps_per_epoch=len(self.generator),
                                 epochs=100,
                                 callbacks=[eva])


def named_entity_recognize(text, model, CRF, id2class):
    """命名实体识别函数
    """
    tokens = tokenizer.tokenize(text)
    # print(tokens)
    # print('token', len(tokens))

    while len(tokens) > 512:
        tokens.pop(-2)
    token_ids = tokenizer.tokens_to_ids(tokens)
    segment_ids = [0] * len(token_ids)
    nodes = model.predict([[token_ids], [segment_ids]])[0]
    trans = K.eval(CRF.trans)
    labels = ViterbiDecoder(trans).decode(nodes)[1:-1]
    return labels


def evaluate(data, model, crf, i2tag_dict):
    pre_list = []
    true_list = []
    for d in tqdm(data):
        text = ''.join(d['content'][:510])
        true = d['tag'][:510]
        r = named_entity_recognize(text, model, crf, i2tag_dict)
        r = [i2tag_dict.get(str(i)) for i in r]
        pre_list.append(r)
        print('pred:', r)
        true_list.append(true)
        print("True:", true)
        print('_______')
    a, p, r, f = get_ner_fmeasure(true_list, pre_list, label_type="BIO")
    return f, p, r


class Evaluate(keras.callbacks.Callback):
    def __init__(self, model, crf, i2tag_dict, valid, test):
        self.best_val_f1 = 0
        self.model = model
        self.CRF = crf
        self.i2tag_dict = i2tag_dict
        self.valid_data = valid
        self.test_data = test

    def on_epoch_end(self, epoch, logs=None):
        trans = K.eval(self.CRF.trans)
        f1, precision, recall = evaluate(self.valid_data, self.model, self.CRF, self.i2tag_dict)
        # 保存最优
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            self.model.save_weights('./best_model.weights')
        print('valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
              (f1, precision, recall, self.best_val_f1))
        # f1, precision, recall = evaluate(self.test_data, self.model, self.CRF, self.i2tag_dict)
        # print('test:  f1: %.5f, precision: %.5f, recall: %.5f\n' %
        #       (f1, precision, recall))


if __name__ == '__main__':
    ROOT_PATH = '/content/drive/My Drive/'
    path = ROOT_PATH + 'CCKS_2017/data/raw_data/data2.jsonl'
    data = read_jsonline(path)
    random.shuffle(data)

    tag2i_dict = {'O': 0,
                  'B-TREATMENT': 1,
                  'I-TREATMENT': 2,
                  'B-BODY': 3,
                  'I-BODY': 4,
                  'B-SIGNS': 5,
                  'I-SIGNS': 6,
                  'B-CHECK': 7,
                  'I-CHECK': 8,
                  'B-DISEASE': 9,
                  'I-DISEASE': 10}

    i2tag_dict = {str(v): k for k, v in tag2i_dict.items()}
    max_len = max([len(i['content']) for i in data])
    num = int(len(data) * 0.8)

    train_data = data[:num]
    val_data = data[num:num + 50]
    test_data = data[num:]
    generator = Data_generator(train_data, dict_path, tag2i_dict, 512)
    v_generator = Data_generator(val_data, dict_path, tag2i_dict, 512)
    tokenizer = generator.tokenizer
    electra_crf = PretrainCrf(dict_path, config_path, checkpoint_path, generator)
    eva = Evaluate(electra_crf.model, electra_crf.CRF, i2tag_dict, val_data, test_data)
    electra_crf.train(eva)



