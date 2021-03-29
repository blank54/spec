#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
from config import Config
with open('/data/blank54/workspace/project/spec/spec.cfg', 'r') as f:
    cfg = Config(f)

sys.path.append(cfg['root'])
from analysis import *
utils = Utils()


def load_w2v_model():
    iter_unit = 'paragraph'
    parameters = {'size': 200, 'window': 10, 'iter': 200, 'min_count': 10, 'workers': 4, 'sg': 1, 'hs': 1, 'negative': 5, 'ns_exponent': 0.75}
    
    fdir_model = os.path.join(cfg['root'], cfg['fdir_w2v_model'])
    fname_w2v_model = '{}_ngram_{}.pk'.format(iter_unit, utils.parameters2fname(parameters))
    fpath_w2v_model = os.path.join(fdir_model, fname_w2v_model)
    
    with open(fpath_w2v_model, 'rb') as f:
        w2v_model = pk.load(f)

    return w2v_model

def eval_w2v_model(w2v_model):
    vocabs = sorted(w2v_model.model.wv.vocab)
    print('# of Vocabs: {:,}'.format(len(vocabs)))


if __name__ == '__main__':
    w2v_model = load_w2v_model()
    eval_w2v_model(w2v_model=w2v_model)