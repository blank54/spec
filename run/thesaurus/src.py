#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
import pickle as pk
from time import time

from gensim.models import Word2Vec

from config import Config
with open('/data/blank54/workspace/project/spec/spec.cfg', 'r') as f:
    cfg = Config(f)

sys.path.append(cfg['root'])
from analysis import *
read = Read()
write = Write()
utils = Utils()
visualizer = Visualizer()


## Word2Vec Embedding
def word2vec_embedding(iter_unit, train):
    parameters = {'size': 200, 'window': 10, 'iter': 200, 'min_count': 10, 'workers': 4, 'sg': 1, 'hs': 1, 'negative': 5, 'ns_exponent': 0.75}
    fdir_model = os.path.join(cfg['root'], cfg['fdir_w2v_model'])
    fname_w2v_model = '{}_ngram_{}.pk'.format(iter_unit, utils.parameters2fname(parameters))
    fpath_w2v_model = os.path.join(fdir_model, fname_w2v_model)

    _start = time()

    if train:
        docs = [d.ngram for d in read.docs(iter_unit=iter_unit)]
        model = Word2Vec(
            size=parameters.get('size'),
            window=parameters.get('window'),
            iter=parameters.get('iter'),
            min_count=parameters.get('min_count'),
            workers=parameters.get('workers'),
            sg=parameters.get('sg'),
            hs=parameters.get('hs'),
            negative=parameters.get('negative'),
            ns_exponent=parameters.get('ns_exponent'),
        )
        w2v_model = Word2VecModel(docs=docs, model=model, parameters=parameters)
        w2v_model.train()
        write.object(obj=w2v_model, fpath=fpath_w2v_model)
    else:
        with open(fpath_w2v_model, 'rb') as f:
            w2v_model = pk.load(f)

    _end = time()
    print('Training Word2Vec Model [{}]: {:,.02f} minutes'.format(iter_unit, (_end-_start)/60))
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
    fdir_word_flows = os.path.join(cfg['root'], cfg['fdir_word_map'])
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
    fdir_rules = os.path.join(cfg['root'], cfg['fdir_word_map'])
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
    w2v_model = word2vec_embedding(iter_unit='paragraph', train=True)
    # evaluate_w2v_model(w2v_model)
    
    word_flows = calculate_word_flow(min_similarity=0.7, do=True, w2v_model=w2v_model)
    # visualize_word_flow(word_flows)

    word_mapping_rules = get_mapping_rules(do=True, word_flows=word_flows)
    # visualize_word_map(word_mapping_rules)