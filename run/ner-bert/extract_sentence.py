#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import nltk
import pickle as pk
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.append('/data/blank54/workspace/project/spec/')
from specutil import SpecPath
from analysis import Read
specpath = SpecPath()
read = Read()


def explore_data():
    data = []
    for sent in read.docs(iter_unit='sentence'):
        sent_len = len(sent.text.split())
        data.append(sent_len)

    plt.hist(data, bins=len(set(data)))
    plt.show()


def develop_corpus():
    print('____________________________________________________________')
    
    cnt = 0
    sent_len_range = range(10, 50)

    corpus = []
    for idx, sent in enumerate(read.docs(iter_unit='sentence')):
        tokens = sent.text.split()

        ## The length of sentences should be in sent_len_range.
        if len(tokens) not in sent_len_range:
            continue

        ## Sentences should include verb.
        sent.pos = nltk.pos_tag(tokens)
        if any([('V' in tag) for token, tag in sent.pos]):
            corpus.append(sent)

        ## Verbose
        cnt = idx
        if idx % 500 == 0:
            sys.stdout.write('\rDeveloping Corpus from {:,} sentences'.format(idx))

    with open(specpath.corpus_sentence, 'wb') as f:
        pk.dump(corpus, f)

    print('\n____________________________________________________________')
    print('Sentence Corpus')
    print('  | Initial Sentences: {:,}'.format(cnt))
    print('  | Corpus Length:     {:,}'.format(len(corpus)))
    print('  | Saved at: "{}"'.format(specpath.corpus_sentence))
    print('____________________________________________________________')


if __name__ == '__main__':
    # explore_data()
    develop_corpus()