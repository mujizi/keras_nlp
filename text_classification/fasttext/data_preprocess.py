# -*- coding: utf-8 -*-
# @Time    : 2020/5/19 下午5:35
# @Author  : Benqi

import pandas as pd
import re
from constant import ROOT_PATH
from sklearn.model_selection import train_test_split


def filter_content(s):
    s = re.sub('\{IMG:.?.?.?\}', '', s)                    #图片
    s = re.sub(re.compile(r'[a-zA-Z]+://[^\s]+'), '', s)   #网址
    s = re.sub(re.compile('<.*?>'), '', s)                 #网页标签
    s = re.sub(re.compile('&[a-zA-Z]+;?'), ' ', s)         #网页标签
    s = re.sub(re.compile('[a-zA-Z0-9]*[./]+[a-zA-Z0-9./]+[a-zA-Z0-9./]*'), ' ', s)
    s = re.sub("\?{2,}", "", s)
    s = re.sub("\r", "", s)
    s = re.sub("\n", ",", s)
    s = re.sub("\t", ",", s)
    s = re.sub("（", ",", s)
    s = re.sub("）", ",", s)
    s = re.sub("\u3000", "", s)
    s = re.sub(" ", "", s)
    r4 = re.compile('\d{4}[-/]\d{2}[-/]\d{2}')             #日期
    s=re.sub(r4,'某时',s)
    return s


if __name__ == '__main__':
    toutiao_path = ROOT_PATH + 'data/toutiao/toutiao_cat_data.txt'
    dest_train = ROOT_PATH + 'data/toutiao/toutiao_train.csv'
    dest_test = ROOT_PATH + 'data/toutiao/toutiao_test.csv'
    with open(toutiao_path, 'r', encoding='utf-8') as f:
        x_l, y_l = [], []
        for i in f.readlines():
            i = filter_content(i)
            i_list = i.split("_")
            if i_list[7] != '!':
                x_l.append(i_list[7])
                y_l.append(i_list[5])
        x_train, x_test, y_train, y_test = train_test_split(x_l, y_l, test_size=0.2)
        toutiao_train = pd.DataFrame({'x': x_train, 'y': y_train})
        toutiao_test = pd.DataFrame({'x': x_test, 'y': y_test})


    toutiao_train.to_csv(dest_train)
    toutiao_test.to_csv(dest_test)
    print(toutiao_train.loc[:, 'y'].value_counts())
    print(toutiao_test.loc[:, 'y'].value_counts())
