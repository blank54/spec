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
embedding = Embedding()



if __name__ == '__main__':
    ## Build NER Corpus
    fname_ner_corpus = 'ner_corpus.pk'
    ner_corpus_init = BuildCorpus().ner_corpus(max_sent_len=50, fname=fname_ner_corpus, build=True)

    ## Word2Vec Embedding
    fname_w2v_model_old = 'paragraph_ngram_200_10_200_10_4_1_1_5_0.75.pk'
    docs_for_w2v_update = [d.words for d in ner_corpus_init.docs]
    feature_size, word_vector = embedding.update_word2vec(fname=fname_w2v_model_old, docs=docs_for_w2v_update, update=True)
    ner_corpus = embedding.ner_word_embedding(ner_corpus=ner_corpus_init, feature_size=feature_size, word_vector=word_vector)