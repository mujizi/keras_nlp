# -*- coding: utf-8 -*-
# @Time    : 2020/5/28 下午5:37
# @Author  : Benqi

import codecs
import numpy as np
from keras.optimizers import Adam
from tools import read_jsonline
from constant import dict_path, config_path, checkpoint_path
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras.layers import Input, Dense
from keras import Model
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint


"""本文件有bug"""
def seq_padding(x, pad_len, value=0):
    return np.array([np.concatenate([i, [value] * (pad_len - len(i))]) if len(i) < pad_len else i[:pad_len] for i in x])


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]') # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]') # 剩余的字符是[UNK]
        return R


class Data_generator:
    def __init__(self, data, dict_path, label_dict, max_len, batch_size=200):
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
                x1, x2 = self.tokenizer.encode(first=text)
                x_l1.append(x1)
                x_l2.append(x2)
                y_l.append(tag)
                if self.batch_size == len(x_l1) or idxs[-1] == i:
                    X1 = seq_padding(x_l1, self.max_len)
                    X2 = seq_padding(x_l2, self.max_len)
                    y_index = [[self.label_dict.get(j) for j in i] for i in y_l]
                    y_padding = seq_padding(y_index, self.max_len, self.label_dict['O'])
                    y_one_hot = np.array([to_categorical(i, len(self.label_dict)) for i in y_padding])
                    yield [X1, X2], y_one_hot
                    x_l1, x_l2, y_l = [], [], []


class BertCrf:
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
        output = Dense(11, use_bias=False, activation='softmax')(x)
        crf = CRF(11)
        output = crf(output)
        bert_crf = Model([input1, input2], output)
        bert_crf.compile(loss=crf_loss,
                         optimizer=Adam(lr=1e-5),
                         metrics=[crf_viterbi_accuracy, 'accuracy'])
        bert_crf.summary()
        return bert_crf

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
                                 validation_data=self.v_generator.__iter__(),
                                 nb_val_samples=200)

    def predict(self):
        pass


if __name__ == '__main__':
    ROOT_PATH = '/Users/ouhon/PycharmProjects/keras_nlp_tutorial/NER/'
    path = ROOT_PATH + 'CCKS_2017/data/raw_data/data.jsonl'
    data = read_jsonline(path)

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

    i2tag_dict = {str(v): str(i) for i, v in tag2i_dict.items()}
    max_len = max([len(i['content']) for i in data])
    num = int(len(data) * 0.8)
    train_data = data[:num]
    val_data = data[num:]
    generator = Data_generator(train_data, dict_path, tag2i_dict, 512)
    v_generator = Data_generator(val_data, dict_path, tag2i_dict, 512)
    [x1, x2], y = generator.__iter__().__next__()

    # bert_crf = BertCrf(dict_path, config_path, checkpoint_path, generator, v_generator)
    # bert_crf.train()
    print(x1.shape)
    print(y.shape)

