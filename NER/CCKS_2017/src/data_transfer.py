# -*- coding: utf-8 -*-
# @Time    : 2020/5/28 下午3:05
# @Author  : Benqi
import os
from tools import write_jsonline


class DataTransfer:
    def __init__(self):
        cur = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        self.raw_data_path = os.path.join(os.path.join(cur, os.path.pardir), 'data/raw_data')
        self.train_path = os.path.join(os.path.join(cur, os.path.pardir), 'data/raw_data/data.jsonl')
        self.cn2en_dict = {
                      '检查和检验': 'CHECK',
                      '症状和体征': 'SIGNS',
                      '疾病和诊断': 'DISEASE',
                      '治疗': 'TREATMENT',
                      '身体部位': 'BODY'}

        self.tag2i_dict = {
            'O': 0,
            'B-TREATMENT': 1,
            'I-TREATMENT': 2,
            'B-BODY': 3,
            'I-BODY': 4,
            'B-SIGNS': 5,
            'I-SIGNS': 6,
            'B-CHECK': 7,
            'I-CHECK': 8,
            'B-DISEASE': 9,
            'I-DISEASE': 10
        }

    def single_file2jsonl(self, path):
        filepath = path
        label_filepath = filepath.replace('.txtoriginal', '')
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            with open(label_filepath, 'r', encoding='utf') as l_f:
                res = l_f.readlines()
                tag_l = ['O'] * len(content)
                content = [i for i in content]
                for r in res:
                    r = r.strip().split('\t')
                    tag, s_i, e_i, feature = r[0], int(r[1]), int(r[2]), r[3]

                    feature = self.cn2en_dict.get(feature)
                    for i in range(s_i, e_i + 1):
                        if i == s_i:
                            tag_l[i] = 'B-' + feature
                        else:
                            tag_l[i] = 'I-' + feature
                assert len(tag_l) == len(content)
                res_dic = {'content': content, 'tag': tag_l}
        return res_dic

    def transfer(self):
        list_dir = os.listdir(self.raw_data_path)[1:]
        res_jsonl = []
        for dir in list_dir:
            dir_path = self.raw_data_path + '/' + dir
            files = os.listdir(dir_path)
            for file in files:
                file_path = dir_path + '/' + file
                if 'original' not in file_path:
                    continue
                res_dic = self.single_file2jsonl(file_path)
                res_jsonl.append(res_dic)
        write_jsonline(self.train_path, res_jsonl)


if __name__ == '__main__':
    data_trans = DataTransfer()
    data_trans.transfer()





