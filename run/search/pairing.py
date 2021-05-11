#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(1) #Do not print INFO
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(2)

import sys
import pandas as pd

from config import Config
with open('/data/blank54/workspace/project/spec/spec.cfg', 'r') as f:
    cfg = Config(f)

sys.path.append(cfg['root'])
from model import BERT


def import_data():
    fpath = '/data/blank54/workspace/project/spec/data/provision_pair/test/test.xlsx'
    data = pd.read_excel(fpath, na_values=1)
    return data


def load_bert_weights(model):
    bert_model = bert.fine_tuning(model=model)

    fdir_bert_trained = os.path.join(cfg['root'], cfg['fdir_provision_pairing_bert'])
    fname_bert_model = 'bert.h5'
    fpath_bert_model = os.path.join(fdir_bert_trained, fname_bert_model)
    bert_model.load_weights(filepath=fpath_bert_model)

    return bert_model


if __name__ == '__main__':
    ## Call BERT
    bert_dist = 'uncased_L-12_H-768_A-12'
    fdir_bert_pretrained = os.path.join(cfg['root'], cfg['fdir_bert_pretrained'], bert_dist)
    bert = BERT(fdir_pretrained=fdir_bert_pretrained, SEQ_LEN=128)

    ## Data Preparation
    token_dict, _ = bert.build_vocab()
    data = import_data()

    data_X = bert.data2input(token_dict=token_dict, data=data, option='predict')
    
    ## Load BERT Model
    pretrained_model = bert.load_model()
    bert_model = load_bert_weights(model=pretrained_model)

    ## Predict
    print(bert_model.predict(data_X))