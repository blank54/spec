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
    ner_parameters = {
        'lstm_units': 512,
        'lstm_return_sequences': True,
        'lstm_recurrent_dropout': 0.2,
        'dense_units': 50,
        'dense_activation': 'relu',
        'test_size': 0.3,
        'batch_size': 256, #32,
        'epochs': 1,
        'validation_split': 0.1,
    }

    ## NER Corpus
    fname_ner_corpus = 'ner_corpus.pk'
    ner_corpus = read.ner_corpus(fname=fname_ner_corpus)

    ## NER Model Identification
    fname_ner_model = 'ner_model.h5'
    ner_model = NER_Model()
    ner_model.initialize(ner_corpus=ner_corpus, parameters=ner_parameters)
    
    ## NER Model Training
    X = ner_corpus.X_embedded
    Y = ner_corpus.Y_embedded
    ner_model.train(X=X, Y=Y, parameters=ner_parameters)
    ner_model.save(fname=fname_ner_model)

    ## Load
    ner_model = NER_Model(fname=fname_ner_model)
    ner_model.load(ner_corpus=ner_corpus, parameters=ner_parameters)
    ner_model.evaluate()