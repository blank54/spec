#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
import itertools

from config import Config
with open('/data/blank54/workspace/project/spec/spec.cfg', 'r') as f:
    cfg = Config(f)

sys.path.append(cfg['root'])
from analysis import *
corpus = Corpus()
preprocessor = Preprocessor()


## Data Import
fdir_sents = os.path.join(cfg['root'], cfg['fdir_corpus'], 'sentence')
fpath_sents = os.path.join(fdir_sents, 'sentences_original.pk')
sents_original = corpus.load_single(fpath_sents)
print('# of Original Corpus: {}'.format(len(sents_original)))


## Text Cleaning
fpath_sents_cleaned = os.path.join(fdir_sents, 'sentences_cleaned.pk')
sents_cleaned = preprocessor.text_cleaning(sents=sents_original, 
                                           fpath=fpath_sents_cleaned, 
                                           do=False)
print('Size of Original: {:.02f} KB'.format(os.path.getsize(fpath_sents)/1024))
print('Size of Cleaned: {:.02f} KB'.format(os.path.getsize(fpath_sents_cleaned)/1024))


## TODO
## Normalization
fpath_sents_normalized = os.path.join(fdir_sents, 'sentences_normalized.pk')
sents_normalized, normalized_tokens = preprocessor.normalize(sents=sents_cleaned, 
                                                             fpath=fpath_sents_normalized,
                                                             do=False)
print('# of Sentences with Replaced Tokens: {}'.format(len(sents_normalized)))
for t in normalized_tokens.keys():
    print(' └ # of Sentences Operated with {}: {}'.format(t, len(normalized_tokens[t])))


## Bullet Points Adjustment
# for s in sents:
#     if s.text.startswith('?')


## PoS Tagging -> Extract real sents
fpath_sents_verbal = os.path.join(fdir_sents, 'sentences_verbal.pk')
sents_verbal = preprocessor.extract_sent_verb(sents=sents_normalized, 
                                              fpath=fpath_sents_verbal, 
                                              do=False)
print('# of Original Sents: {}'.format(len(sents_normalized)))
print('# of Sents with Verb: {}'.format(len(sents_verbal)))


## Stopword Removal
'''
Refer to other studies that utilized Word2Vec
'''
fpath_sents_wo_stop = os.path.join(fdir_sents, 'sentences_wo_stop.pk')
sents_wo_stop = preprocessor.stopword_removal(sents=sents_verbal,
                                              fpath=fpath_sents_wo_stop,
                                              do=False)
print('# of Sents without Stopwords: {}'.format(len(sents_wo_stop)))
print(' └ # of Words BEFORE Stopword Removal: {}'.format(len(list(itertools.chain(*[s.pos for s in sents_verbal])))))
print(' └ # of Words AFTER Stopword Removal : {}'.format(len(list(itertools.chain(*[s.pos for s in sents_wo_stop])))))


## Text Chunking
fpath_sents_chunk = os.path.join(fdir_sents, 'sentences_chunk.pk')
sents_chunk = preprocessor.chunking(sents=sents_wo_stop,
                                    fpath=fpath_sents_chunk,
                                    do=False)
print('# of Chunks: {}'.format(len(list(itertools.chain(*[s.chunk for s in sents_chunk])))))