# -*- coding: utf-8 -*-
# @Time    : 2020/6/18 下午8:13
# @Author  : Benqi
from collections import Counter
from intervaltree import IntervalTree
from itertools import accumulate
from NER.CCKS_2020_task3.src.utils.io import *


def entity2label_batch(paragraph_list, label_schema='BILOU'):
    """
    transform entities and text in paragraph list to labels
    :param paragraph_list: paragraph list
    :param label_schema: label schema, only allow 'BILOU', 'BIO' and None.
                        When its value is none, label will be uppercase of entity type
    :return:
    """
    for item in paragraph_list:
        entity2label(item, label_schema)
    return paragraph_list


def entity2label(data, label_schema='BILOU', resolve_conflict=True):
    """
    transform text, entities, token and pos tags into labels, labels will attach in data object
    :param data: source data
    :param label_schema: label schema, only allow 'BILOU', 'BIO' and None.
                        When its value is none, label will be uppercase of entity type
    :param resolve_conflict: whether resolve conflict which entity start or end is in token intermediate position
    :return: new data dict with labels
    """
    if 'text' not in data:
        raise KeyError('must have text field')
    elif 'entities' not in data:
        raise KeyError('muse have entities field')
    elif 'tokens' not in data:
        raise KeyError('must have tokens field')

    text = data['text']
    entities = data['entities']
    tokens = data['tokens']

    if resolve_conflict:
        tokens = split_conflict_spans(entities, tokens, text)

    token_spans = [(t['start'], t['end']) for t in tokens]
    token_starts, token_ends = list(zip(*token_spans))
    labels = ['O'] * len(token_spans)

    if len(set([s for s, e in token_spans])) < len(token_spans):
        dup = Counter([s for s, e in token_spans]).most_common(1)
        print(dup)
        raise Exception('have duplicate span')

    tree = IntervalTree.from_tuples(token_spans)
    for entity in entities:
        entity_type = entity['type']
        if not entity_type:
            raise ValueError('entity type is empty')
        start_intervals = list(tree.at(entity['start']))
        if len(start_intervals) != 1:
            raise Exception('start interval count error')
        end_intervals = list(tree.at(entity['end'] - 1))
        if len(end_intervals) != 1:
            raise Exception('end interval count error')
        start_index = token_starts.index(start_intervals[0].begin)
        end_index = token_ends.index(end_intervals[0].end) + 1
        if label_schema == 'BILOU':
            if end_index - start_index == 1:
                labels[start_index] = join_label('U', entity_type)
            else:
                inter_label = join_label('I', entity_type)
                if end_index - start_index > 2:
                    labels[start_index + 1:end_index - 1] = [inter_label] * (end_index - start_index - 2)
                labels[start_index] = join_label('B', entity_type)
                labels[end_index - 1] = join_label('L', entity_type)
        elif label_schema == 'BIO':
            labels[start_index] = join_label('B', entity_type)
            if end_index - start_index > 1:
                inter_label = join_label('I', entity_type)
                labels[start_index + 1:end_index] = [inter_label] * (end_index - start_index - 1)
        elif not label_schema:
            labels[start_index:end_index] = [entity['type'].upper()] * (end_index - start_index)
        else:
            raise Exception('label schema is not supported.')

    data['tokens'] = tokens
    data['labels'] = labels

    return data


def join_label(label, entity_type):
    return label + '-' + entity_type


def split_conflict_spans(entities, tokens, text):
    """
    resolve the conflict in tokens and entities
    :param entities: entity list
    :param tokens: token list, everyone is a dict which has text, start, end and pos tag (optional)
    :param text: original text
    :return: new tokens whose conflicts have been resolved.
    """
    if not tokens:
        return tokens
    token_spans = [(t['start'], t['end']) for t in tokens]
    token_starts, token_ends = list(zip(*token_spans))
    split_indices = []
    for e in entities:
        if e['start'] not in token_starts:
            split_indices.append(e['start'])
        if e['end'] not in token_ends:
            split_indices.append(e['end'])
    split_indices = sorted(set(split_indices))
    for split_index in split_indices:
        tree = IntervalTree.from_tuples(token_spans)
        print(tree)
        interval = list(tree.at(split_index))
        print(interval)
        # print(split_index)
        print(len(interval))
        if len(interval) != 1:
            raise Exception('interval count error, {}'.format(len(interval)))
        interval = interval[0]
        if interval.begin == split_index:
            raise Exception('split index start error')
        elif interval.end == split_index:
            raise Exception('split index end error')
        token_spans.extend([(interval.begin, split_index), (split_index, interval.end)])
        if (interval.begin, interval.end) in token_spans:
            token_spans.remove((interval.begin, interval.end))

    token_spans = sorted(token_spans, key=lambda s: s[0])
    new_tokens = [{'text': text[s:e], 'start': s, 'end': e} for (s, e) in token_spans]

    if 'pos_tag' in tokens[0]:
        pos_tags = []
        token_starts, token_ends = list(zip(*token_spans))
        for token in tokens:
            pos_tag = token['pos_tag']
            start = token['start']
            end = token['end']
            pos_start = token_starts.index(start)
            pos_end = token_ends.index(end)
            if pos_start == pos_end:
                pos_tags.append(pos_tag)
            elif pos_end - pos_start > 0:
                pos_tags.extend([pos_tag] * (pos_end - pos_start + 1))
            else:
                raise ValueError('token end index is before token start index')
        pos_tag_count = len(pos_tags)
        new_token_count = len(new_tokens)
        msg = 'pos tag count and token text count are not equal, token {}, pos tag {}'
        if pos_tag_count != new_token_count:
            raise Exception(msg.format(new_token_count, pos_tag_count))
        else:
            for token, pos_tag in zip(new_tokens, pos_tags):
                token['pos_tag'] = pos_tag
    return new_tokens


if __name__ == '__main__':
    path = "/Users/ouhon/PycharmProjects/keras_nlp_tutorial/NER/CCKS_2020_task3/dataset/task1_train.jsonl"
    file = read_jsonline(path)
    l = entity2label_batch(file, "BIO")