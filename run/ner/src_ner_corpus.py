#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys

from config import Config
with open('/data/blank54/workspace/project/spec/spec.cfg', 'r') as f:
    cfg = Config(f)

sys.path.append(cfg['root'])
from analysis import BuildCorpus, Read, Write
buildcorpus = BuildCorpus()
read = Read()
write = Write()


def update_word2vec(docs, update):
    fdir = os.path.join(cfg['root'], cfg['fdir_w2v_model'])
    fname_old = 'paragraph_ngram_200_10_200_10_4_1_1_5_0.75.pk'
    fpath_old = os.path.join(fdir, fname_old)
    fname_new = fname_old.replace('.pk', '_update.pk')
    fpath_new = os.path.join(fdir, fname_new)

    if update:
        w2v_model = read.object(format='pk', fpath=fpath_old)
        w2v_model.train(docs=docs_for_w2v_update, update=update)
        write.object(obj=w2v_model, fpath=fpath_new)
    else:
        w2v_model = read.object(format='pk', fpath=fpath_new)

    return w2v_model


if __name__ == '__main__':
    ## Build NER Corpus
    fname_ner_corpus = 'ner_corpus.pk'
    ner_corpus = BuildCorpus().ner_corpus(max_sent_len=50, fname=fname_ner_corpus, build=False)
    docs_for_w2v_update = [d.words for d in ner_corpus.docs]

    ## Word2Vec Embedding
    w2v_model = update_word2vec(docs=docs_for_w2v_update, update=True)
    ner_corpus.word_embedding(embedding_model=w2v_model)
    write.object(obj=ner_corpus, fpath=ner_corpus.fpath)