# -*- coding: utf-8 -*-
# @Time    : 2020/6/30 下午10:32
# @Author  : Benqi

from NER.CCKS_2020_task3.src.utils.io import read_jsonline, write_jsonline


def cut_sentence(text):
    sign_list, cut_list = [], []
    for index, t in enumerate(text):
        if t in ['。', '：']:
            sign_list.append(index)
    # print(sign_list)
    start, end = 0, 0
    for index, i in enumerate(sign_list):
        if (i - start > 510) and (end - start < 510):
            cut_list.append(sign_list[index - 1])
            start = sign_list[index - 1]
        elif (i - start > 510) and (end - start > 510):
            cut_list.append(sign_list[index - 2])
            start = sign_list[index - 2]
        end = i
    return cut_list


def batch_cut(data, dest_path):
    new = []
    for i in data:
        text = i['text']
        label = i['label']
        cut_index = cut_sentence(text)
        if cut_index != []:
            if len(cut_index) == 1:
                dic['text'] =
                new_text.append(text[:cut_index[0] + 1])
                new_text.append(text[cut_index[0] + 1:])

            start = 0
            for i in range(0, len(cut_index)):
                if i + 1 <= len(cut_index):
                    new_text.append(text[start:i + 1])
                    start = i + 1
                else:
                    new_text.append(text[i + 1:])
        else:
            new_text.append(text)
    print(new_text)


if __name__ == '__main__':
    path = '/Users/ouhon/PycharmProjects/keras_nlp_tutorial/NER/CCKS_2020_task3/dataset/entities2label.jsonl'
    dest_path = '/Users/ouhon/PycharmProjects/keras_nlp_tutorial/NER/CCKS_2020_task3/dataset/cut_sen.jsonl'
    data = read_jsonline(path)
    batch_cut(data[:10], dest_path)
