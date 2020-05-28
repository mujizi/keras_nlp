# -*- coding: utf-8 -*-
# @Time    : 2020/5/27 下午2:19
# @Author  : Benqi

import pandas as pd
from tools import write_jsonline


def get_dataset(raw_path, dest_path):
    df = pd.read_csv(raw_path, encoding = "ISO-8859-1", error_bad_lines=False)
    dataset = df.drop(['Unnamed: 0', 'lemma', 'next-lemma', 'next-next-lemma', 'next-next-pos',
                       'next-next-shape', 'next-next-word', 'next-pos', 'next-shape',
                       'next-word', 'prev-iob', 'prev-lemma', 'prev-pos',
                       'prev-prev-iob', 'prev-prev-lemma', 'prev-prev-pos', 'prev-prev-shape',
                       'prev-prev-word', 'prev-shape', 'prev-word', 'shape'], axis=1)
    print(dataset.head())
    dataset.to_csv(dest_path)
    return dataset


class SentenceGetter(object):

    def __init__(self, dataset):
        self.n_sent = 1
        self.dataset = dataset
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["word"].values.tolist(),
                                                           s['pos'].values.tolist(),
                                                           s["tag"].values.tolist())]
        self.grouped = self.dataset.groupby("sentence_idx").apply(agg_func)
        print(self.dataset.groupby("sentence_idx"))
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


def build_jsonline(sentences, dest_path):
    res = []
    for sentence in sentences:
        w_l = [i[0] for i in sentence]
        p_l = [i[1] for i in sentence]
        t_l = [i[2] for i in sentence]
        dic = {'word': w_l, "pos": p_l, "tag": t_l}
        res.append(dic)
    write_jsonline(dest_path, res)


if __name__ == '__main__':
    raw_path = "/NER/lstm_crf/data/raw_data/ner.csv"
    dest_path = "/NER/lstm_crf/data/raw_data/dataset.csv"
    dest_jsonl = "/Users/ouhon/PycharmProjects/keras_nlp_tutorial/NER/data/raw_data/dataset.jsonl"
    dataset = get_dataset(raw_path, dest_path)
    getter = SentenceGetter(dataset)
    sentences = [i for i in getter.sentences]
    build_jsonline(sentences, dest_jsonl)