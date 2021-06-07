#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
import itertools
from tqdm import tqdm

from config import Config
with open('/data/blank54/workspace/project/spec/spec.cfg', 'r') as f:
    cfg = Config(f)

sys.path.append(cfg['root'])
from analysis import *
preprocessor = Preprocessor()
# ngram_parser = NgramParser()
read = Read()
write = Write()


def preprocess(iter_unit, do):
    size_before = 0
    size_after = 0

    unit_map = read.word_map(option='unit')
    stopword_list = read.stopword_list()

    fdir_corpus = os.path.join(cfg['root'], cfg['fdir_corpus'], iter_unit)
    with tqdm(total=len(os.listdir(fdir_corpus))) as pbar:
        for d in read.docs(iter_unit=iter_unit):
            size_before += os.path.getsize(d.fpath)

            if do:
                text_normalized = preprocessor.normalize(text=d.text, unit_map=unit_map)
                d.token = preprocessor.tokenize(text=text_normalized)
                d.stop = preprocessor.stopword_removal(sent=d.token, stopword_list=stopword_list)
                d.stem = preprocessor.stemmize(sent=d.stop)
                d.lemma = preprocessor.lemmatize(sent=d.stop)
                write.object(obj=d, fpath=d.fpath)
            else:
                pass

            size_after += os.path.getsize(d.fpath)
            pbar.update(1)

    print('Preprocessing [{}]\n └ {:,.02f} MB -> {:,.02f} MB'.format(iter_unit, size_before/1024**2, size_after/1024**2))


## Ngram
def ngram_parsing(iter_unit, do):
    size_before = 0
    size_after = 0

    if do:
        docs = [d.lemma for d in read.docs(iter_unit=iter_unit)]
        bigram_counter = ngram_parser.count(docs=docs)
    else:
        pass

    fdir_corpus = os.path.join(cfg['root'], cfg['fdir_corpus'], iter_unit)
    with tqdm(total=len(os.listdir(fdir_corpus))) as pbar:
        for d in read.docs(iter_unit=iter_unit):
            size_before += os.path.getsize(d.fpath)

            if do:
                d.ngram = ngram_parser.parse(sent=d.lemma, bigram_counter=bigram_counter)
                write.object(obj=d, fpath=d.fpath)
            else:
                pass

            size_after += os.path.getsize(d.fpath)
            pbar.update(1)

        print('Ngram Parsing [{}]:\n └ {:,.02f} MB -> {:,.02f} MB'.format(iter_unit, size_before/1024**2, size_after/1024**2))





if __name__ == '__main__':
    # preprocess(iter_unit='section_manual', do=False)
    # ngram_parsing(iter_unit='section_manual', do=False)

    # preprocess(iter_unit='paragraph', do=True)
    # ngram_parsing(iter_unit='paragraph', do=True)

    preprocess(iter_unit='sentence', do=True)