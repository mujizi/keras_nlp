# -*- coding: utf-8 -*-
# @Time    : 2020/6/18 下午6:44
# @Author  : Benqi

from NER.CCKS_2020_task3.src.utils.highlight import *


def data_review(filename, name2id, lang='cn'):
    if lang not in {'en', 'cn'}:
        raise Exception('language is not supported')
    data_list = read_jsonline(filename)
    for data in data_list[:30]:
        text = data['originalText']
        spans = [(e['start_pos'], e['end_pos'], name2id[e['label_type']]) for e in data['entities']]
        print(highlight_by_spans(text, spans), '\n')


if __name__ == '__main__':
    name2id = {"疾病和诊断":"black", "影像检查":"purple", "实验室检验":"green", "手术":"blue", "药物":"yellow", "解剖部位":"red"}
    path = '/Users/ouhon/PycharmProjects/keras_nlp_tutorial/NER/CCKS_2020_task3/dataset/task1_train.jsonl'
    data_review(path, name2id)
