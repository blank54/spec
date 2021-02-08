#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
from time import time

from config import Config
with open('/data/blank54/workspace/project/spec/spec.cfg', 'r') as f:
    cfg = Config(f)

sys.path.append(cfg['root'])
from analysis import *
corpus = Corpus()
utils = Utils()
stat = Stat()


## Data Import
fdir_corpus = os.path.join(cfg['root'], cfg['fdir_corpus'])
fpath_sents = os.path.join(fdir_corpus, 'manual/sentence', 'sentences_chunk.pk')
sents = corpus.load_single(fpath=fpath_sents)


## Word Embedding
for size in [50, 100, 200]:
    for window in [1, 2, 3, 5, 10, 20]:
        for iter_num in [100, 200, 500, 1000]:
            for min_count in [0, 3, 5, 10, 50, 100]:
                for negative in [0, 1, 3, 5, 10]:
                    fdir_model = os.path.join(cfg['root'], cfg['fdir_model'])
                    parameters = {
                        'size': size,
                        'window': window,
                        'iter': iter_num,
                        'min_count': min_count,
                        'workers': 4,
                        'sg': 1,
                        'hs': 1,
                        'negative': negative,
                        'ns_exponent': 1,
                    }
                    fname_w2v_model = '{}.pk'.format(utils.parameters2fname(parameters))
                    fpath_w2v_model = os.path.join(fdir_model, 'w2v', fname_w2v_model)

                    _start = time()
                    w2v_model = Embedding().word2vec(fpath=fpath_w2v_model, 
                                                     sents=sents, 
                                                     parameters=parameters, 
                                                     train=True)
                    _end = time()
                    print('Training Word2Vec Model: {:,.02f} minutes'.format((_end-_start)/60))
                    print('# of Vocabs: {}'.format(len(w2v_model.model.wv.vocab)))


## Evaluation of Embedding
# word_list = ['aashto-NUM/JJ', 'astm-NUM/JJ']
# for idx, w in enumerate(word_list):
# # for idx, w in enumerate(list(w2v_model.model.wv.vocab)[:100]):
#     print('{}: {}'.format(w, ', '.join([w for w, s in w2v_model.model.wv.most_similar(w, topn=5)])))


## Flow Margin
# words = {}
# for word in w2v_model.model.wv.vocab:
#     similar_list = [(w, s) for (w, s) in w2v_model.model.wv.most_similar(word, topn=10) if s>=0.7]
#     if similar_list:
#         words[word] = similar_list

# for word in sorted(words.keys()):
#     print('{}: {}'.format(word, ', '.join(['{}({:.03f})'.format(w, s) for w, s in words[word]])))


## Pivot Term Determination




## Term Map