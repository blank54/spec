#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
import codecs
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from config import Config
with open('/data/blank54/workspace/project/spec/spec.cfg', 'r') as f:
    cfg = Config(f)

sys.path.append(cfg['root'])
from analysis import Read, BERT_Tokenizer
read = Read()


def build_vocab(fpath_vocab):
    token_dict = {}
    with codecs.open(fpath_vocab, 'r', encoding='utf-8') as reader:
        for line in reader:
            token = line.strip()
            if '_' in token:
                token = token.replace('_', '')
                token = '##' + token
            token_dict[token] = len(token_dict)
    init_len = len(token_dict)

    for doc in read.docs(iter_unit='paragraph'):
        for domain_token in [t.lower() for t in doc.token]:
            if domain_token not in token_dict.keys():
                token = '##' + domain_token
                token_dict[token] = len(token_dict)
            else:
                continue
    updated_len = len(token_dict)
    
    print('Initial vocabs: {}'.format(init_len))
    print('Updated vocabs: {}'.format(updated_len))
    reverse_dict = {i: t for t, i in token_dict.items()}
    return token_dict, reverse_dict

def build_tokenizer(token_dict):
    return BERT_Tokenizer(token_dict)

def import_data(TARGET_TAG, REFER_TAG):
    fname = 'L-{}_R-{}.xlsx'.format(TARGET_TAG, REFER_TAG)
    fdir = os.path.join(cfg['root'], cfg['fdir_data_provision_pair_labeled'])
    fpath = os.path.join(fdir, fname)

    data = pd.read_excel(fpath)
    print('Read data "{}" from "../{}"'.format(fname, cfg['fdir_data_provision_pair_labeled']))
    train, test = train_test_split(data, train_size=0.7, shuffle=True)
    return train, test

def data2input(data, option='train'):
    global tokenizer
    
    data[LEFT_COLUMN] = data[LEFT_COLUMN].astype(str)
    data[RIGHT_COLUMN] = data[RIGHT_COLUMN].astype(str)

    indices, targets = [], []
    for idx in tqdm(range(len(data))):
        ids, segments = tokenizer.encode(data[LEFT_COLUMN].iloc[idx], data[RIGHT_COLUMN].iloc[idx], max_len=SEQ_LEN)
        indices.append(ids)
        targets.append(data[LABEL_COLUMN].iloc[idx])
        
    X = [np.array(indices), np.zeros_like(indices)]
    Y = np.array(targets)
    
    if option=='train' or option=='test':
        return X, Y
    elif option=='predict':
        return X
    else:
        print('ERROR: Wrong option!!!')
        return None

def load_bert(bert_dist, SEQ_LEN):
    model = read.bert_pretrained(bert_dist=bert_dist, SEQ_LEN=SEQ_LEN)
    return model



if __name__ == '__main__':
    TARGET_TAG = 'qatar_2014_06_05'
    REFER_TAG = 'qatar_2010_06_05'
    
    LEFT_COLUMN = 'left_text'
    RIGHT_COLUMN = 'right_text'
    LABEL_COLUMN = 'label'

    bert_dist = 'uncased_L-12_H-768_A-12'
    SEQ_LEN = 128 #maximum length of input sentence
    fpath_pretrained_bert = os.path.join(cfg['root'], cfg['fdir_bert_pretrained'], bert_dist)
    fpath_vocab = os.path.join(fpath_pretrained_bert, 'vocab.txt')

    # token_dict, reverse_dict = build_vocab(fpath_vocab)
    # tokenizer = build_tokenizer(token_dict)

    # train, test = import_data(TARGET_TAG, REFER_TAG)
    # train_X, train_Y = data2input(data=train, option='train')
    # test_X, test_Y = data2input(test, option='test')

    model = load_bert(bert_dist=bert_dist, SEQ_LEN=SEQ_LEN)
    model.summary()