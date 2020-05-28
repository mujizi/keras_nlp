# -*- coding: utf-8 -*-
# @Time    : 2020/5/27 下午2:52
# @Author  : Benqi
from tools import read_jsonline, write_json


def get_dic(path, t2i_path, l2i_path):
    sentences = read_jsonline(path)
    words = set(word for i in sentences for word in i['word'])
    token_dic = {str(v): i + 1 for i, v in enumerate(words)}
    token_dic["unk"] = 0
    labels = set(word for i in sentences for word in i['tag'])
    label2id = {str(v): i for i, v in enumerate(labels)}
    write_json(t2i_path, token_dic)
    write_json(l2i_path, label2id)


if __name__ == '__main__':
    path = "/NER/lstm_crf/data/raw_data/dataset.jsonl"
    token_dic_path = "/NER/lstm_crf/data/raw_data/token2i.json"
    labels_dic_path = "/NER/lstm_crf/data/raw_data/label2i.json"
    get_dic(path, token_dic_path, labels_dic_path)