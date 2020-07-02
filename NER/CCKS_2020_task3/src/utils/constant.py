# -*- coding: utf-8 -*-
# @Time    : 2020/6/18 下午6:33
# @Author  : Benqi
import os

__code_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
__base_dir = os.path.join(__code_dir, os.pardir)
BASE_DIR = os.path.realpath(__base_dir) + '/'
DATA_DIR = BASE_DIR + 'data/'
MODEL_DIR = DATA_DIR + 'models/'
EVALUATION_DIR = DATA_DIR + 'evaluation/'
TRAINING_CONLL_FILE = DATA_DIR + 'training.conll'
TRAINING_FILE = DATA_DIR + 'training.json'
VALIDATION_FILE = DATA_DIR + 'validation.json'
VALIDATION_OOV_FILE = DATA_DIR + 'validation_oov.json'
TEST_FILE = DATA_DIR + 'test.json'
TEST_OOV_FILE = DATA_DIR + 'test_oov.json'
CONFIG_DIR = BASE_DIR + 'config/'
BRAT_CONFIG_DIR = CONFIG_DIR + 'brat/'

# patent section names
PATENT_SECTIONS = ['title', 'abstract', 'claim', 'description']

NGRAM_DELIMITER = ' '
EMPTY_PLACEHOLDER = '\u3000'

# evaluation
RESULT_MISSING = 'missing'
RESULT_ERROR = 'error'
RESULT_DISPLAY = 'display'
RESULT_MORE = 'more'
DISPLAY_MODE = {RESULT_ERROR, RESULT_MISSING, RESULT_DISPLAY}
DEFAULT_COLOR = 'yellow'

# name
VAR_TRAINING_FILE = 'training_file'
VAR_TRAINING_OOV_FILE = 'training_oov_file'
VAR_VALIDATION_FILE = 'validation_file'
VAR_VALIDATION_OOV_FILE = 'validation_oov_file'
VAR_TEST_FILE = 'test_file'
VAR_TEST_OOV_FILE = 'test_oov_file'

# experiment parameter

# feature names
UNIGRAM = 'UNIGRAM'
UNIGRAM_PREV_1 = 'UNIGRAM:-1'
UNIGRAM_NEXT_1 = 'UNIGRAM:1'
BIGRAM = 'BIGRAM'

# type
DEFAULT_TYPE = 'part_id'
ENTITY_TYPES = ['part_id', 'part_name']

# CoNLL tags
SEQ_BIO = 'BIO'
SEQ_BILOU = 'BILOU'
DEFAULT_LABELS = 'BILOU'