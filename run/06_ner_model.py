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
read = Read()
write = Write()
utils = Utils()
embedding = Embedding()








if __name__ == '__main__':
    fname_ner_corpus = 'ner_corpus.pk'
    ner_corpus = read.ner_corpus(fname=fname_ner_corpus)

    ## NER Model Development
    ner_model_parameters = {
        'lstm_units': 512,
        'lstm_return_sequences': True,
        'lstm_recurrent_dropout': 0.2,
        'dense_units': 50,
        'dense_activation': 'relu',
    }
    ner_model = NER_Modeling().initialize(ner_corpus=ner_corpus, parameters=ner_model_parameters)
    ## TODO