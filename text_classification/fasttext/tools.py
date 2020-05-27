# -*- coding: utf-8 -*-
# @Time    : 2020/5/20 下午1:14
# @Author  : Benqi

import os
import json
from collections import Iterable
import configparser

__ENCODING_UTF8 = 'utf-8'


def read_lines(filename, encoding=__ENCODING_UTF8, *, strip=True, filter_empty=True, default=None):
    """
    read lines in text file
    :param filename: file path
    :param encoding: encoding of the file, default is utf-8
    :param strip: whether strip every line, default is True
    :param filter_empty: whether filter empty line, when strip is True, judge after strip
    :param default: default value to return if file is not existed. Set it to None to disable it.
    :return: lines
    """
    if default is not None and not os.path.exists(filename):
        return default

    with open(filename, encoding=encoding) as f:
        if strip:
            if filter_empty:
                return [l.strip() for l in f.read().splitlines() if l.strip()]
            else:
                return [l.strip() for l in f.read().splitlines()]
        else:
            if filter_empty:
                return [l for l in f.read().splitlines() if l]
            else:
                return [l for l in f.read().splitlines()]


def read_lines_lazy(src_filename, encoding=__ENCODING_UTF8, *, default=None):
    """
    use generator to load files, one line every time
    :param src_filename: source file path
    :param encoding: file encoding
    :param default: default value to return if file is not existed. Set it to None to disable it.
    :return: lines in file
    """
    if default is not None and not os.path.exists(src_filename):
        return default
    file = open(src_filename, encoding=encoding)
    for line in file:
        yield line
    file.close()


def read_file(filename, encoding=__ENCODING_UTF8):
    """
    wrap open function to read text in file
    :param filename: file path
    :param encoding: encoding of file, default is utf-8
    :return: text in file
    """
    with open(filename, encoding=encoding) as f:
        return f.read()


def write_file(filename, data, encoding=__ENCODING_UTF8):
    """
    write text into file
    :param filename: file path to save
    :param data: text data
    :param encoding: file encoding
    :return: None
    """
    with open(filename, 'w', encoding=encoding) as f:
        f.write(data)


def write_lines(filename, lines, encoding=__ENCODING_UTF8, filter_empty=False, strip=False):
    """
    write lines to file, will add line break for every line automatically
    :param filename: file path to save
    :param lines: lines to save
    :param encoding: file encoding
    :param filter_empty:
    :param strip:
    :return: None
    """
    if isinstance(lines, str):
        raise TypeError('line doesn\'t allow str format')

    if not isinstance(lines, Iterable):
        raise Exception('data can\'t be iterated')

    if strip:
        if filter_empty:
            lines = [l.strip() for l in lines if l.strip()]
        else:
            lines = [l.strip() for l in lines]
    else:
        if filter_empty:
            lines = [l for l in lines if l]

    if not lines:
        raise Exception('lines are empty')

    with open(filename, 'w', encoding=encoding) as f:
        f.write('\n'.join(lines) + '\n')


def read_json(src_filename):
    """
    read json file
    :param src_filename: source file path
    :return: loaded object
    """
    with open(src_filename, encoding=__ENCODING_UTF8) as f:
        return json.load(f)


def write_json(dest_filename, data, serialize_method=None):
    """
    dump json data to file, support non-UTF8 string (will not occur UTF8 hexadecimal code).
    :param dest_filename: destination file path
    :param data: data to be saved
    :param serialize_method: python method to do serialize method
    :return: None
    """
    with open(dest_filename, 'w', encoding=__ENCODING_UTF8) as f:
        if not serialize_method:
            json.dump(data, f, ensure_ascii=False)
        else:
            json.dump(data, f, ensure_ascii=False, default=serialize_method)


def read_jsonline(src_filename, encoding=__ENCODING_UTF8, *, default=None):
    """
    read jsonl file
    :param src_filename: source file path
    :param encoding: file encoding
    :param default: default value to return if file is not existed. Set it to None to disable it.
    :return: object list, an object corresponding a line
    """
    if default is not None and not os.path.exists(src_filename):
        return default
    file = open(src_filename, encoding=encoding)
    items = []
    for line in file:
        items.append(json.loads(line))
    file.close()
    return items


def read_jsonline_lazy(src_filename, encoding=__ENCODING_UTF8, *, default=None):
    """
    use generator to load jsonl one line every time
    :param src_filename: source file path
    :param encoding: file encoding
    :param default: default value to return if file is not existed. Set it to None to disable it.
    :return: json object
    """
    if default is not None and not os.path.exists(src_filename):
        return default
    file = open(src_filename, encoding=encoding)
    for line in file:
        yield json.loads(line)
    file.close()


def write_jsonline(dest_filename, items, encoding=__ENCODING_UTF8):
    """
    write items to file with json line format
    :param dest_filename: destination file path
    :param items: items to be saved line by line
    :param encoding: file encoding
    :return: None
    """
    if isinstance(items, str):
        raise TypeError('json object list can\'t be str')

    if not dest_filename.endswith('.jsonl'):
        print('json line filename doesn\'t end with .jsonl')

    if not isinstance(items, Iterable):
        raise TypeError('items can\'t be iterable')
    file = open(dest_filename, 'w', encoding=encoding)
    for item in items:
        file.write(json.dumps(item, ensure_ascii=False) + '\n')
    file.close()


def read_ini(src_filename):
    """
    read configs in ini file
    :param src_filename: source file path
    :return: parsed config data
    """
    config = configparser.ConfigParser()
    config.read(src_filename)
    return config


def write_ini(dest_filename, config_data):
    """
    write config into file
    :param dest_filename: destination file
    :param config_data: config data
    :return: None
    """
    config = configparser.ConfigParser()
    for key, val in config_data.items():
        config[key] = val
    with open(dest_filename, 'w') as config_file:
        config.write(config_file)


def append_line(dest_filename, line, encoding=__ENCODING_UTF8):
    """
    append single line to file
    :param dest_filename: destination file path
    :param line: line string
    :param encoding: text encoding to save data
    :return: None
    """
    if not isinstance(line, str):
        raise TypeError('line is not in str type')
    with open(dest_filename, 'a', encoding=encoding) as f:
        f.write(line + '\n')


def append_lines(dest_filename, lines, remove_file=False, encoding=__ENCODING_UTF8):
    """
    append lines to file
    :param dest_filename: destination file path
    :param lines: lines to be saved
    :param remove_file: whether remove the destination file before append
    :param encoding: text encoding to save data
    :return:
    """
    if remove_file and os.path.exists(dest_filename):
        os.remove(dest_filename)
    for line in lines:
        append_line(dest_filename, line, encoding)


def append_jsonline(dest_filename, item, encoding=__ENCODING_UTF8):
    """
    append item as a line of json string to file
    :param dest_filename: destination file
    :param item: item to be saved
    :param encoding: file encoding
    :return: None
    """
    with open(dest_filename, 'a', encoding=encoding) as f:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')


def append_jsonlines(dest_filename, items, encoding=__ENCODING_UTF8):
    """
    append item as some lines of json string to file
    :param dest_filename: destination file
    :param items: items to be saved
    :param encoding: file encoding
    :return: None
    """
    with open(dest_filename, 'a', encoding=encoding) as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')