# -*- coding: utf-8 -*-
# @Time    : 2020/6/18 下午6:33
# @Author  : Benqi
import os


def check_entities(entities, text):
    """
    check whether entity object is legal, if error happen, exception will be raised
    :param entities: entity object list
    :param text: original text
    :return: None
    """
    for entity in entities:
        start = entity['start']
        end = entity['end']
        if start < 0 or end < 0:
            raise Exception('offset is negative')
        if text[start:end] != entity['entity']:
            print(entity)
            print(text[start:end])

            raise Exception('entity text and offset don\'t correspond.')


def adjust_entities_offsets(entity_list, offset, start=None, end=None):
    for entity in entity_list:
        not_restrict = not start and not end
        restrict_start = start and start <= entity['start']
        restrict_end = end and end >= entity['end']
        restrict_all = start and end and start <= entity['start'] < entity['end'] <= end
        if not_restrict or restrict_all or restrict_start or restrict_end:
            entity['start'] += offset
            entity['end'] += offset
    return entity_list


def get_filenames_in_folder(folder_name, ext_name=True, hidden_file=False, attach_folder_name=True):
    filenames = []
    if not os.path.exists(folder_name):
        raise Exception('folder is not existed.')
    for filename in os.listdir(folder_name):
        if hidden_file:
            if filename.startswith('.') and filename not in {'.', '..'}:
                filenames.append(filename)
        elif not filename.startswith('.'):
            filenames.append(filename)
    if attach_folder_name:
        filenames = [os.path.join(folder_name, name) for name in filenames]
    if not ext_name:
        filenames = [name[:name.rindex('.')] for name in filenames]
    return filenames


def get_entities_by_type(entity_list, entity_type):
    selected_entities = []

    for entity in entity_list:
        if entity['type'] == entity_type:
            selected_entities.append(entity)

    return selected_entities


def get_index_char2word(tokens, index):
    for idx, token in enumerate(tokens):
        if token['end'] > index:
            return idx
    raise IndexError('index is out of sentence')





def replace_extname(src_filename, new_extname):
    """
    replace extension name
    :param src_filename: source filename
    :param new_extname: new extension name to replace, dot will be appended automatically if not existed
    :return: new filename
    """
    if not new_extname.startswith('.'):
        new_extname = '.' + new_extname
    return os.path.splitext(src_filename)[0] + new_extname