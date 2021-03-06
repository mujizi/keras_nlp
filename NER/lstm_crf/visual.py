# -*- coding: utf-8 -*-
# @Time    : 2020/5/27 下午3:12
# @Author  : Benqi
from tools import read_jsonline
import matplotlib.pyplot as plt


def sentences_len(path):
    sentences = read_jsonline(path)
    len_l = [len(i['word']) for i in sentences]
    print(len_l)
    plt.hist(len_l, bins=40, facecolor="blue", edgecolor="black", alpha=0.7)
    # 显示横轴标签
    plt.xlabel("length")
    # 显示纵轴标签
    plt.ylabel("nums")
    # 显示图标题
    plt.title("statistic")
    plt.show()


if __name__ == '__main__':
    path = "/NER/lstm_crf/data/raw_data/dataset.jsonl"
    sentences_len(path)