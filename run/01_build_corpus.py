#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
import pickle as pk
from tqdm import tqdm

from config import Config
with open('/data/blank54/workspace/project/spec/spec.cfg', 'r') as f:
    cfg = Config(f)

sys.path.append(cfg['root'])
from analysis import *
utils = Utils()

fdir_data = os.path.join(cfg['root'], cfg['fdir_data_section_manual'])
flist = os.listdir(fdir_data)
with tqdm(total=len(flist)) as pbar:
    sentences = []
    for fname in flist:
        fpath = os.path.join(fdir_data, fname)
        sentences.extend(BuildCorpus().section2sentence(fpath_section=fpath)) ## TODO
        pbar.update(1)

fdir_corpus = os.path.join(cfg['root'], cfg['fdir_corpus'])
os.makedirs(fdir_corpus, exist_ok=True)
fname_corpus = 'sentences.pk'
with open(os.path.join(fdir_corpus, fname_corpus), 'wb') as f:
    pk.dump(sentences, f)