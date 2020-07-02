# -*- coding: utf-8 -*-
# @Time    : 2020/6/30 下午3:20
# @Author  : Benqi

from NER.CCKS_2020_task3.src.utils.io import read_jsonline, write_jsonline


def entities2label(jsonl):
    label_dic = {"疾病和诊断": "disease", "影像检查": "image", "实验室检验": "lab", "手术": "surgery", "药物": "medicine", "解剖部位": "body"}
    text = jsonl['originalText']
    entities = jsonl['entities']
    label = len(text) * ["O"]
    jsonl["text"] = [i for i in text]

    for entity in entities:
        s = entity['start_pos']
        e = entity['end_pos']
        type = entity['label_type']
        for i in range(s, e):
            if i == s:
                label[i] = "B-" + label_dic[type]
            else:
                label[i] = "I-" + label_dic[type]
    jsonl["label"] = label
    assert len(jsonl['label']) == len(jsonl['text'])
    return jsonl


def batch_e2l(jsonl_l, dest_path):
    new_data = [entities2label(i) for i in jsonl_l]
    write_jsonline(dest_path, new_data)


if __name__ == '__main__':
    path = '/Users/ouhon/PycharmProjects/keras_nlp_tutorial/NER/CCKS_2020_task3/dataset/task1_train.jsonl'
    dest_path = '/Users/ouhon/PycharmProjects/keras_nlp_tutorial/NER/CCKS_2020_task3/dataset/entities2label.jsonl'
    data = read_jsonline(path)
    batch_e2l(data, dest_path)
