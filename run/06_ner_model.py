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



if __name__ == '__main__':
    ## Parameters
    ner_model_parameters = {
        'lstm_units': 512,
        'lstm_return_sequences': True,
        'lstm_recurrent_dropout': 0.2,
        'dense_units': 50,
        'dense_activation': 'relu',
        'test_size': 0.3,
        'batch_size': 32,
        'epochs': 5,
        'validation_split': 0.1,
    }

    ## NER Corpus
    fname_ner_corpus = 'ner_corpus.pk'
    ner_corpus = read.ner_corpus(fname=fname_ner_corpus)

    ## NER Model Identification
    fname_ner_model = 'ner_model'
    ner_model = NER_Model(fname=fname_ner_model)
    ner_model.initialize(ner_corpus=ner_corpus, parameters=ner_model_parameters)
    
    ## NER Model Training
    X = ner_corpus.X_embedded
    Y = ner_corpus.Y_embedded
    ner_model.train(X=X, Y=Y, parameters=ner_model_parameters)
    # ner_model.save()

    ## Load
    # ner_model = read.ner_model(fname=fname_ner_model)
    # print(ner_model.dense_activation)