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
corpus = Corpus()


## Data Import
fdir_corpus = os.path.join(cfg['root'], cfg['fdir_corpus'])
fpath_sents = os.path.join(fdir_corpus, 'sentence', 'sentences_chunk.pk')


## Word Embedding
docs_for_w2v = 



## Flow Margin




## Pivot Term Determination




## Term Map