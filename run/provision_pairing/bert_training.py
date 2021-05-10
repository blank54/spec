#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(1) #Do not print INFO
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(2)

import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from config import Config
with open('/data/blank54/workspace/project/spec/spec.cfg', 'r') as f:
    cfg = Config(f)

sys.path.append(cfg['root'])
from analysis import BERT


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

def prepare_data_for_training():
    global TARGET_TAG, REFER_TAG, token_dict

    train, test = import_data(TARGET_TAG, REFER_TAG)
    train_X, train_Y = bert.data2input(token_dict=token_dict, data=train, option='train')
    test_X, test_Y = bert.data2input(token_dict=token_dict, data=test, option='test')
    data = [train_X, train_Y, test_X, test_Y]

    return data

def save_bert(model):
    fname = 'bert.h5'
    fdir = os.path.join(cfg['root'], cfg['fdir_provision_pairing_bert'])
    fpath = os.path.join(fdir, fname)
    bert.save_model(model=model, fpath=fpath)


if __name__ == '__main__':
    ## Data Information
    TARGET_TAG = 'qatar_2014_06_05'
    REFER_TAG = 'qatar_2010_06_05'
    
    ## Call BERT
    bert_dist = 'uncased_L-12_H-768_A-12'
    fdir_bert_pretrained = os.path.join(cfg['root'], cfg['fdir_bert_pretrained'], bert_dist)
    bert = BERT(fdir_pretrained=fdir_bert_pretrained, SEQ_LEN=128)

    ## Build Vocabulary
    token_dict, _ = bert.build_vocab()

    ## Data Preparation
    data = prepare_data_for_training()

    ## Train BERT
    pretrained_model = bert.load_model()
    finetuned_model = bert.fine_tuning(model=pretrained_model)

    parameters = {
        'epochs': 100,
        'batch_size': 32,
    }
    trained_model = bert.train(data=data, model=finetuned_model, parameters=parameters)
    bert_model.summary()

    ## Save BERT
    save_bert(model=bert_model)