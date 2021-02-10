#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
import pickle as pk
from time import time

from config import Config
with open('/data/blank54/workspace/project/spec/spec.cfg', 'r') as f:
    cfg = Config(f)

sys.path.append(cfg['root'])
from analysis import *
read = Read()
write = Write()
utils = Utils()
visualizer = Visualizer()


## Word Embedding
def word_embedding(train):
    fdir_model = os.path.join(cfg['root'], cfg['fdir_model'])
    parameters = {
        'size': 200,
        'window': 10,
        'iter': 200,
        'min_count': 10,
        'workers': 4,
        'sg': 1,
        'hs': 1,
        'negative': 5,
        'ns_exponent': 0.75,
    }
    fname_w2v_model = 'paragraph_ngram_{}.pk'.format(utils.parameters2fname(parameters))
    fpath_w2v_model = os.path.join(fdir_model, 'w2v', fname_w2v_model)

    _start = time()

    if train:
        docs = [p.ngram for p in read.docs(iter_unit='paragraph')]
        model = Word2Vec(
            size=parameters['size'],
            window=parameters['window'],
            iter=parameters['iter'],
            min_count=parameters['min_count'],
            workers=parameters['workers'],
            sg=parameters['sg'],
            hs=parameters['hs'],
            negative=parameters['negative'],
            ns_exponent=parameters['ns_exponent'],
        )
        w2v_model = Word2VecModel(docs=docs, model=model, parameters=parameters)
        w2v_model.train()
        write.object(obj=w2v_model, fpath=fpath_w2v_model)
    else:
        with open(fpath_w2v_model, 'rb') as f:
            w2v_model = pk.load(f)

    _end = time()
    print('Training Word2Vec Model: {:,.02f} minutes'.format((_end-_start)/60))
    print('# of Vocabs: {}'.format(len(w2v_model.model.wv.vocab)))
    return w2v_model


## Evaluation of Embedding
def evaluate_w2v_model(w2v_model):
    keywords = ['specification', 'slurry', 'spread', 'smooth', 'include', 'fill', 'exceed', 'contractor']
    for word in keywords: #sorted(w2v_model.model.wv.vocab):
        print('{:15s}: {}'.format(word, ', '.join([w for w, _ in w2v_model.model.wv.most_similar(word)[:5]])))

## Word Flows
def calculate_word_flow(min_similarity, do, **kwargs):
    fname_word_flows = 'word_flows_{}.pk'.format(min_similarity)
    fdir_word_flows = os.path.join(cfg['root'], cfg['fdir_model'], 'thesaurus/')
    fpath_word_flows = os.path.join(fdir_word_flows, fname_word_flows)

    if do:
        w2v_model = kwargs.get('w2v_model')
        word_links = defaultdict(list)        
        for word in w2v_model.model.wv.vocab:
            similar_list = [(w, s) for w, s in w2v_model.model.wv.most_similar(word) if s >= min_similarity]
            word_links[word] = list(sorted(similar_list, key=lambda x:x[1], reverse=True))

        word_flows = []
        for word_to in word_links:
            for rank, (word_from, similarity) in enumerate(word_links[word_to]):
                score = (len(word_links[word_to])-rank)/len(word_links[word_to])
                word_flows.append(WordFlow(word_from=word_from, word_to=word_to, similarity=similarity, score=score))
        
        write.object(obj=word_flows, fpath=fpath_word_flows)

    else:
        with open(fpath_word_flows, 'rb') as f:
            word_flows = pk.load(f)

    print('Word Flow Calculation: {:,} flows'.format(len(word_flows)))
    return word_flows

def visualize_word_flow(word_flows):
    visualizer.network([(f.word_from, f.word_to, f.score) for f in word_flows])


## Pivot Term Determination
def get_mapping_rules(do, **kwargs):
    fname_rules = 'word_mapping_rules.pk'
    fdir_rules = os.path.join(cfg['root'], cfg['fdir_model'], 'thesaurus/')
    fpath_rules = os.path.join(fdir_rules, fname_rules)

    if do:
        word_flows = kwargs.get('word_flows')
        flow_margin = defaultdict(float)
        for flow in word_flows:
            flow_margin[flow.word_from] -= flow.score
            flow_margin[flow.word_to] += flow.score

        rules = {}
        for word in list(flow_margin.keys()):
            if flow_margin[word] > 0:
                rules[word] = word
            else:
                candidates = [f for f in word_flows if f.word_from == word]

                ## TODO: needs to be recursive
                pivot_term = list(sorted(candidates, key=lambda x:flow_margin[x.word_to], reverse=True))[0].word_to
                rules[word] = pivot_term

        write.object(obj=rules, fpath=fpath_rules)

    else:
        with open(fpath_rules, 'rb') as f:
            rules = pk.load(f)

    print('Word Mapping Rules: {:,}'.format(len(rules)))
    return rules

def visualize_word_map(rules):
    visualizer.network([(word_from, word_to, 1) for word_from, word_to in rules.items()])





if __name__ == '__main__':
    w2v_model = word_embedding(train=False)
    # evaluate_w2v_model(w2v_model)
    
    word_flows = calculate_word_flow(min_similarity=0.7, do=False, w2v_model=w2v_model)
    # visualize_word_flow(word_flows)

    word_mapping_rules = get_mapping_rules(do=False, word_flows=word_flows)
    # visualize_word_map(word_mapping_rules)