#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys

from config import Config
with open('/data/blank54/workspace/project/spec/spec.cfg', 'r') as f:
    cfg = Config(f)

sys.path.append(cfg['root'])
from analysis import BERT_Tokenizer


def build_tokenizer(fpath_vocab):
    token_dict = {}
    with codecs.open(vocab_path, 'r', encoding='utf-8') as reader:
        for line in reader:
            token = line.strip()
            if '_' in token:
                token = token.replace('_', '')
                token = '##' + token
            token_dict[token] = len(token_dict)

    tokenizer = BERT_Tokenizer(token_dict)
    return tokenizer

def data2input(data, option='train'):
    global tokenizer
    
    data[DATA_COLUMN] = data[DATA_COLUMN].astype(str)
    indices, targets = [], []
    for idx in tqdm(range(len(data))):
        ids, segments = tokenizer.encode(data[DATA_COLUMN][idx], max_len=SEQ_LEN)
        indices.append(ids)
        targets.append(data[LABEL_COLUMN][idx])
        
    X = [np.array(indices), np.zeros_like(indices)]
    Y = np.array(targets)
    
    if option=='train' or option=='test':
        return X, Y
    elif option=='predict':
        return X
    else:
        print('ERROR: Wrong option!!!')
        return None



if __name__ == '__main__':
    SEQ_LEN = 128 #maximum length of input sentence
    BATCH_SIZE = 16
    EPOCHS = 2
    LR = 1e-5

    bert_dist = 'uncased_L-12_H-768_A-12'
    fpath_pretrained_bert = os.path.join(cfg['root'], cfg['fdir_bert_pretrained'], bert_dist)
    fpath_bert_config = os.path.join(fpath_pretrained_bert, 'bert_config.json')
    fpath_bert_checkpoint = os.path.join(fpath_pretrained_bert, 'bert_model.ckpt')
    fpath_bert_vocab = os.path.join(fpath_pretrained_bert, 'vocab.txt')

    DATA_COLUMN = 'provision_pair'
    LABEL_COLUMN = 'label'

    tokenizer = build_tokenizer(fpath_vocab)
    train_X, train_Y = data2input(train, option='train')
    test_X, test_Y = data2input(test, option='test')

    print(train_X)