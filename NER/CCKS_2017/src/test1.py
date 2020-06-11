# -*- coding: utf-8 -*-
# @Time    : 2020/6/5 下午2:24
# @Author  : Benqi
from tools import read_jsonline, write_jsonline


new = []
f = read_jsonline('/Users/ouhon/PycharmProjects/keras_nlp_tutorial/NER/CCKS_2017/data/raw_data/data.jsonl')
for i in f:
    l = i['tag']
    if len(set(l)) > 1:
        new.append(i)
        print(i)
write_jsonline("/Users/ouhon/PycharmProjects/keras_nlp_tutorial/NER/CCKS_2017/data/raw_data/data2.jsonl", new)
print(len(f))
print(len(new))
