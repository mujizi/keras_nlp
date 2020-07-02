# -*- coding: utf-8 -*-
# @Time    : 2020/6/18 下午6:30
# @Author  : Benqi
from collections import OrderedDict
from itertools import accumulate
from .io import read_json, read_jsonline
from .constant import ENTITY_TYPES, DEFAULT_COLOR
from .utils import *


HIGHLIGHT_COLOR = OrderedDict([
    ('grey', '0;35;47m'),
    ('yellow', '0;30;43m'),
    ('blue', '0;30;44m'),
    ('black', '3;37;40m'),
    ('green', '0;37;42m'),
    ('red', '1;31;40m'),
    ('purple', '0:30:45m')
])


def data_review(filename, skip_empty=True, lang='cn'):
    if lang not in {'en', 'cn'}:
        raise Exception('language is not supported')
    data1 = read_jsonline(filename)
    data = data1[0]

    print(data['entities'])
    spans = [(e['start_pos'], e['end_pos'], e['label_type']) for e in data['entities']]
    # if not spans and skip_empty:
    #     continue
    print('===========================')
    data['tokens'] = [str(i) for i in data['originalText']]
    print(highlight_by_spans(data['originalText'], spans))


def get_entity_type_color_mapper(entity_types=ENTITY_TYPES):
    if len(entity_types) > len(HIGHLIGHT_COLOR):
        raise ValueError('entity type count is larger than upper bound.')
    color_mapper = {}
    for entity_type, color_name in zip(entity_types, HIGHLIGHT_COLOR):
        color_mapper[entity_type] = color_name
    if len(color_mapper) <= 1:
        return None
    else:
        return color_mapper


ENTITY_TYPE_COLOR_MAPPER = get_entity_type_color_mapper()


def highlight_data(data, skip_empty=True, in_token=False, sep_str='===================='):
    if isinstance(data, dict):
        highlight_paragraph(data, in_token)
    else:
        for sent in data:
            if skip_empty and not sent['entities']:
                continue
            print(sep_str)
            print(highlight_paragraph(sent, in_token))


def highlight_paragraph(paragraph, in_token=False):
    if not isinstance(paragraph, dict):
        raise TypeError('paragraph is not in dict')
    if 'text' not in paragraph or 'entities' not in paragraph:
        raise ValueError('paragraph doesn\'t have text or entities key')
    spans = []
    if not ENTITY_TYPE_COLOR_MAPPER:
        spans = [(e['start'], e['end']) for e in paragraph['entities']]
    else:
        for entity in paragraph['entities']:
            if entity['type'] in ENTITY_TYPE_COLOR_MAPPER:
                color = ENTITY_TYPE_COLOR_MAPPER[entity['type']]
            else:
                color = DEFAULT_COLOR
            span = (entity['start'], entity['end'], color)
            spans.append(span)

    if not in_token:
        return highlight_by_spans(paragraph['text'], spans)
    else:
        return highlight_by_spans_with_tokens(paragraph['tokens'], spans)


def highlight(s, color):
    """
    add background color of text to highlight it
    :param s: text to highlight
    :param color: color name, must be in `HIGHLIGHT_COLOR dict`
    :return: highlighted text, only work in ipython environment
    """
    return "\033[" + HIGHLIGHT_COLOR[color] + s + "\033[0m"


def highlight_by_spans(sentence, spans, default_color=DEFAULT_COLOR):
    """
    highlight sentence by spans
    :param sentence: sentence text to be highlighted
    :param spans: spans list, span list item is start,end and color index tuple,
                    color is optional, such as (1, 10), (1, 11, 'red')
    :param default_color: default color name
    :return: highlighted text
    """
    spans = merge_spans(spans)
    if not spans:
        return sentence
    display_sentence = sentence[:spans[0][0]]
    next_start = [span[0] for span in spans[1:]] + [len(sentence)]
    for span, next_s in zip(spans, next_start):
        if len(span) == 2:
            s, e = span
            color = default_color
        elif len(span) == 3:
            s, e, color = span
        else:
            hint = 'span tuple count is error. allowed 2 or 3, actual {0}'.format(len(span))
            raise ValueError(hint)
        if e < len(sentence):
            display_sentence += highlight(sentence[s:e], color) + sentence[e:next_s]
        else:
            display_sentence += highlight(sentence[s:e], color)

    return display_sentence


def highlight_by_spans_with_tokens(tokens,
                                   spans,
                                   default_color=DEFAULT_COLOR,
                                   is_merge_spans=False):
    """
    highlight tokens by spans, used for non whitespace split language, such as Chinese, Japanese
    :param tokens: token list, ** original text is ''.join(tokens) **
    :param spans: highlight span list in original text,
                    span list item is start,end and color index tuple,
                    color is optional, such as (1, 10), (1, 11, 'red')
    :param default_color: default color name
    :param is_merge_spans: whether merge spans,
                            It should be False when two entities will be adjacent.
    :return: highlighted text
    """
    if is_merge_spans:
        spans = merge_spans(spans)
    if not spans:
        return ' '.join([token['text'] for token in tokens])

    tokens = list(tokens)
    sentence_len = tokens[-1]['end']
    offsets = [token['start'] for token in tokens] + [sentence_len]
    tokens = inject_tokens_by_outlier_spans(offsets, spans, tokens)
    offsets = [t['start'] for t in tokens] + [sentence_len]
    token_texts = [token['text'] for token in tokens]

    tok_delimiter = ' '
    next_starts = [span[0] for span in spans[1:]] + [sentence_len]
    highlight_text = tok_delimiter.join(token_texts[:offsets.index(spans[0][0])]) + tok_delimiter
    for span, next_start in zip(spans, next_starts):
        if len(span) == 2:
            start, end = span
            color = default_color
        elif len(span) == 3:
            start, end, color = span
        else:
            hint = 'span tuple count is error. allowed 2 or 3, actual {0}'.format(len(span))
            raise ValueError(hint)
        tok_start_idx = offsets.index(start)
        tok_end_idx = offsets.index(end)
        tok_next_start_idx = offsets.index(next_start)
        entity_with_token = tok_delimiter.join(token_texts[tok_start_idx:tok_end_idx])
        highlight_entity = highlight(entity_with_token, color)
        if end < sentence_len:
            next_text = tok_delimiter + tok_delimiter.join(token_texts[tok_end_idx:tok_next_start_idx]) + tok_delimiter
            highlight_text += highlight_entity + next_text
        else:
            highlight_text += highlight_entity

    return highlight_text


def inject_tokens_by_outlier_spans(offsets, spans, tokens):
    ext_spliters = []
    end = tokens[-1]['end']
    for span in spans:
        s, e = span[0], span[1]
        if s < 0 or e < 0:
            raise Exception('span position is less than zero')
        if s not in offsets:
            ext_spliters.append(s)
        if e != end and e not in offsets:
            ext_spliters.append(e)
    spliter_mapper = OrderedDict()
    for ext_spliter in ext_spliters:
        token_idx = get_index_char2word(tokens, ext_spliter)
        if token_idx not in spliter_mapper:
            spliter_mapper[token_idx] = [ext_spliter]
        else:
            spliter_mapper[token_idx].append(ext_spliter)
    replace_items = []
    for token_idx, spliters in spliter_mapper.items():
        ext_tokens = []
        token = tokens[token_idx]
        token_text = token['text']
        start = token['start']
        for spliter in spliters:
            rel_offset = spliter - start
            rel_text = token_text[start - token['start']:rel_offset]
            ext_tokens.append({'text': rel_text, 'start': start, 'end': rel_offset})
            start = spliter
        ext_tokens.append({'text': token_text[start:], 'start': start, 'end': token['end']})
        replace_items.append((token_idx, ext_tokens))
    tokens = replace_item_in_list(tokens, replace_items, True)
    return tokens


def merge_spans(spans):
    """
    merge spans with overlaps, ensure every position is in 1 span at most.
    :param spans: span list
    :return: merged span list without overlaps
    """
    if not spans:
        return spans
    spans = sorted(spans, key=lambda i: i[0])
    new_spans = [spans[0]]
    for span in spans[1:]:
        if len(span) == 2:
            s, e = span
            color = None
        elif len(span) == 3:
            s, e, color = span
        else:
            hint = 'span tuple count is error. allowed 2 or 3, actual {0}'.format(len(span))
            raise ValueError(hint)
        last = new_spans[-1]
        if last[1] < s:
            if not color:
                new_spans.append((s, e))
            else:
                new_spans.append((s, e, color))
        else:
            if s >= last[0] and e <= last[1]:
                continue
            else:
                if not color:
                    new_spans[-1] = (last[0], e)
                else:
                    new_spans[-1] = (last[0], e, color)

    return new_spans