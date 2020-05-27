# -*- coding: utf-8 -*-
# @Time    : 2020/5/20 下午2:47
# @Author  : Benqi
import pandas as pd
from tools import write_json
from constant import ROOT_PATH
import jieba


def create_ngram_set(input_list, ngram_value=2):
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def add_ngram(sequences, token_indice, ngram_range=2):
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for i in range(len(new_list) - ngram_range + 1):
            ngram = tuple(new_list[i:i + 2])
            if ngram in token_indice:
                new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences


def token_dict2json(path, token_dict_path, label2i_path):
    """用来生成中文词对应索引字典，和label的对应索引字典"""
    train_data = pd.read_csv(path)
    labels = train_data['y']
    label2i = {str(i): index for index, i in enumerate(set(labels))}
    i2label = {str(value): key for key, value in label2i.items()}
    print(i2label)
    write_json(label2i_path, label2i)
    train_data_x = [jieba.lcut(i) for i in train_data['x']]
    char_set = set(word for sen in train_data_x for word in sen)
    char_dic = {str(j): i + 1 for i, j in enumerate(char_set)}
    char_dic["unk"] = 0
    print(len(char_dic))
    max_features = len(char_dic)
    new_sequential = [[char_dic.get(word) for word in sen] for sen in train_data_x]
    ngram_range = 2
    print('Adding {}-gram features'.format(ngram_range))

    ngram_set = set()
    for input_list in new_sequential:
        set_of_ngram = create_ngram_set(input_list, ngram_value=2)
        ngram_set.update(set_of_ngram)

    start_index = max_features
    token_indice = {str(v): k + start_index for k, v in enumerate(ngram_set)}
    token_dict = {**token_indice, **char_dic}
    write_json(token_dict_path, token_dict)


if __name__ == '__main__':
    path = ROOT_PATH + 'data/toutiao/toutiao_train.csv'
    token_dict_path = ROOT_PATH + 'data/toutiao/token_dict.json'
    label2i_path = ROOT_PATH + 'data/toutiao/label2i.json'
    token_dict2json(path, token_dict_path, label2i_path)
