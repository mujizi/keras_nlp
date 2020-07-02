# -*- coding: utf-8 -*-
# @Time    : 2020/6/30 下午2:45
# @Author  : Benqi
from NER.CCKS_2020_task3.src.utils.io import read_jsonline
import matplotlib.pyplot as plt


def visual_sen_len(data):
    len_l = [len(i['originalText']) for i in data]
    plt.hist(len_l, bins=10, facecolor="blue", edgecolor="black", alpha=0.7)
    # 显示横轴标签
    plt.xlabel("length")
    # 显示纵轴标签
    plt.ylabel("nums")
    # 显示图标题
    plt.title("statistic")
    print(len(data))
    plt.show()


if __name__ == '__main__':
    path = "/Users/ouhon/PycharmProjects/keras_nlp_tutorial/NER/CCKS_2020_task3/dataset/task1_train.jsonl"
    data = read_jsonline(path)
    visual_sen_len(data)