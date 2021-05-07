#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(1) #Do not print INFO
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(2)

import sys
import json
import codecs
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import keras
from keras import Input, Model
from keras.optimizers import Adam
from keras.layers import Dense

from config import Config
with open('/data/blank54/workspace/project/spec/spec.cfg', 'r') as f:
    cfg = Config(f)

sys.path.append(cfg['root'])
from analysis import Read, Write, BERT_Tokenizer
read = Read()
write = Write()


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

    # for doc in read.docs(iter_unit='paragraph'):
    #     for domain_token in [t.lower() for t in doc.token]:
    #         if domain_token not in token_dict.keys():
    #             token = '##' + domain_token
    #             token_dict[token] = len(token_dict)
    #         else:
    #             continue
    updated_len = len(token_dict)
    
    print('Build BERT Vocabs')
    print('    | Initial: {}'.format(init_len))
    print('    | Updated: {}'.format(updated_len))
    reverse_dict = {i: t for t, i in token_dict.items()}
    return token_dict, reverse_dict

def build_tokenizer(token_dict):
    tokenizer = BERT_Tokenizer(token_dict)
    return tokenizer

def import_data(TARGET_TAG, REFER_TAG):
    fname = 'L-{}_R-{}.xlsx'.format(TARGET_TAG, REFER_TAG)
    fdir = os.path.join(cfg['root'], cfg['fdir_data_provision_pair_labeled'])
    fpath = os.path.join(fdir, fname)

    data = pd.read_excel(fpath)
    print('Load Data for Training')
    print('    | fdir : ../{}'.format(cfg['fdir_data_provision_pair_labeled']))
    print('    | fname: {}'.format(fname))
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

        target = [0]*5 #
        target[data[LABEL_COLUMN].iloc[idx]] = 1
        targets.append(target)
        
    X = [np.array(indices), np.zeros_like(indices)]
    Y = np.array(targets)
    
    if option=='train' or option=='test':
        return X, Y
    elif option=='predict':
        return X
    else:
        print('ERROR: Wrong option!!!')
        return None

def load_pretrained_model(show_summary):
    global bert_dist, SEQ_LEN, tokenizer

    # fpath_pretrained_bert = os.path.join(cfg['root'], cfg['fdir_bert_pretrained'], bert_dist)
    # fpath_bert_config = os.path.join(fpath_pretrained_bert, 'bert_config.json')
    # with open(fpath_bert_config, 'r', encoding='utf-8') as f:
    #     config = json.load(f)
    #     config['vocab_size'] = len(token_dict)
    #     write.json(obj=config, fpath=fpath_bert_config)

    model = read.bert_pretrained(bert_dist=bert_dist, SEQ_LEN=SEQ_LEN)
    print(type(model))
    # model.resize_token_embeddings(len(tokenizer))

    if show_summary:
        model.summary()
    else:
        pass

    return model

def fine_tuning(model, show_summary):
    global SEQ_LEN

    inputs = model.inputs[:2]
    dense = model.layers[-3].output
    outputs = Dense(
        units=5,
        activation='sigmoid',
        name='ProvisionPairLabel'
    )(dense)

    bert_model = Model(inputs=inputs, outputs=outputs)
    bert_model.compile(
        optimizer=Adam(learning_rate=0.00001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # print(bert_model.outputs)
    if show_summary:
        bert_model.summary()
    else:
        pass

    return bert_model

def train_bert(model):
    global bert_dist, train_X, train_Y, test_X, test_Y
    history = model.fit(train_X, train_Y,
                        epochs=100,
                        batch_size=16,
                        verbose=1,
                        validation_data=(test_X, test_Y),
                        shuffle=True)

    fname = 'bert.h5'
    fdir = os.path.join(cfg['root'], cfg['fdir_provision_pairing_bert'])
    fpath = os.path.join(fdir, fname)
    model.save_weights(fpath)
    print('Trained BERT-based ProvisionPairing Model:')
    print('    | fdir : {}'.format(fdir))
    print('    | fname: {}'.format(fname))


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

    token_dict, reverse_dict = build_vocab(fpath_vocab)
    tokenizer = build_tokenizer(token_dict)

    train, test = import_data(TARGET_TAG, REFER_TAG)
    train_X, train_Y = data2input(data=train, option='train')
    test_X, test_Y = data2input(data=test, option='test')

    model = load_pretrained_model(show_summary=True)
    bert_model = fine_tuning(model=model, show_summary=True)
    train_bert(model=bert_model)