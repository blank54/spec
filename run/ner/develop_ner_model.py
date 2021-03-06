#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys

from config import Config
with open('/data/blank54/workspace/project/spec/spec.cfg', 'r') as f:
    cfg = Config(f)

sys.path.append(cfg['root'])
from model import NER_Model
from analysis import Read
read = Read()


## Develop Model
def train_ner_model(ner_parameters, fname_ner_corpus, fname_ner_model):
    ner_corpus = read.ner_corpus(fname=fname_ner_corpus)
    ner_model = NER_Model()
    ner_model.initialize(ner_corpus=ner_corpus, parameters=ner_parameters)

    X = ner_corpus.X_embedded
    Y = ner_corpus.Y_embedded
    ner_model.train(X=X, Y=Y, parameters=ner_parameters)
    ner_model.evaluate()

    ner_model.save(fname=fname_ner_model)

def load_ner_model(ner_parameters, fname_ner_corpus, fname_ner_model):
    ner_corpus = read.ner_corpus(fname=fname_ner_corpus)
    ner_model = NER_Model(fname=fname_ner_model)
    ner_model.load(ner_corpus=ner_corpus, parameters=ner_parameters)
    ner_model.evaluate()
    return ner_model

def ner_predict(ner_model, text):
    sent = text.split()
    ner_result = ner_model.predict(sent=sent)
    print(ner_result)


## Run
if __name__ == '__main__':
    ner_parameters = {
        'lstm_units': 512,
        'lstm_return_sequences': True,
        'lstm_recurrent_dropout': 0.2,
        'dense_units': 100,
        'dense_activation': 'relu',
        'test_size': 0.3,
        'batch_size': 32,
        'epochs': 200,
        'validation_split': 0.1,
    }

    fname_ner_corpus = 'ner_corpus.pk'
    fname_ner_model = 'ner_model.h5'
    text = 'The design and quality control of ACHM surface course mix shall be according to Section 404'
    
    train_ner_model(ner_parameters, fname_ner_corpus, fname_ner_model)
    ner_model = load_ner_model(ner_parameters, fname_ner_corpus, fname_ner_model)
    ner_predict(ner_model=ner_model, text=text)