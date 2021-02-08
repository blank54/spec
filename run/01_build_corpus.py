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


def build_corpus_section_manual():
    fdir_data = os.path.join(cfg['root'], cfg['fdir_data_section_manual'])
    flist = os.listdir(fdir_data)

    sentences = []
    with tqdm(total=len(flist)) as pbar:
        for fname in flist:
            fpath = os.path.join(fdir_data, fname)
            with open(fpath, 'r', encoding='utf-8') as f:
                section = re.sub('\n+\n', '\n\n', f.read().replace('\ufeff', ''))

            sentences.extend(BuildCorpus().section2sentence(data=section))
            pbar.update(1)

    fdir_corpus = os.path.join(cfg['root'], cfg['fdir_corpus'], 'sentence_manual/')
    os.makedirs(fdir_corpus, exist_ok=True)
    fname_sentences = 'sentences_original.pk'
    fpath_sentences = os.path.join(fdir_corpus, fname_sentences)
    with open(fpath_sentences, 'wb') as f:
        pk.dump(sentences, f)

    print('Build Corpus from ManualSection:\n └ {}'.format(fpath_sentences))

def build_corpus_spec():
    fdir_data = os.path.join(cfg['root'], cfg['fdir_data_spec'], 'txt/')
    flist = os.listdir(fdir_data)

    sentences = []
    with tqdm(total=len(flist)) as pbar:
        for fname in flist:
            info = utils.fname2spec_info(fname)
            fpath = os.path.join(fdir_data, fname)
            with open(fpath, 'r', encoding='utf-8') as f:
                spec = f.read()

            sentences.extend(BuildCorpus().spec2sentence(info=info, data=spec, min_sent_len=10))
            pbar.update(1)

    fdir_corpus = os.path.join(cfg['root'], cfg['fdir_corpus'], 'sentence/')
    os.makedirs(fdir_corpus, exist_ok=True)
    fname_sentences = 'sentences_original.pk'
    fpath_sentences = os.path.join(fdir_corpus, fname_sentences)
    with open(fpath_sentences, 'wb') as f:
        pk.dump(sentences, f)

    print('Build Corpus from Spec:\n └ {}'.format(fpath_sentences))


## Run here
# build_corpus_section_manual()
build_corpus_spec()