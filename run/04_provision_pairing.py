#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
import pickle as pk
import pandas as pd
from time import time
import itertools
from collections import defaultdict

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument


from config import Config
with open('/data/blank54/workspace/project/spec/spec.cfg', 'r') as f:
    cfg = Config(f)

sys.path.append(cfg['root'])
from analysis import *
from object import *
read = Read()
write = Write()
utils = Utils()


## Doc2Vec Embedding
def doc2vec_embedding(iter_unit, parameters, train, **kwargs):
    fdir_model = os.path.join(cfg['root'], cfg['fdir_model'])
    fname_model = '{}_ngram_{}.pk'.format(iter_unit, utils.parameters2fname(parameters))
    fpath_model = os.path.join(fdir_model, 'd2v', fname_model)    

    _start = time()

    if train:
        docs = kwargs.get('docs', [TaggedDocument(words=d.ngram, tags=[d.tag]) for d in read.docs(iter_unit=iter_unit)])
        model = Doc2Vec(
            vector_size=parameters.get('vector_size'),
            window=parameters.get('window'),
            min_count=parameters.get('min_count'),
            workers=parameters.get('workers'),
            dm=parameters.get('dm'),
            negative=parameters.get('negative'),
            epochs=parameters.get('epochs'),
            dbow_words=parameters.get('dbow_words'),
        )
        d2v_model = Doc2VecModel(docs=docs, model=model, parameters=parameters)
        d2v_model.train()
        write.object(obj=d2v_model, fpath=fpath_model)
    else:
        with open(fpath_model, 'rb') as f:
            d2v_model = pk.load(f)

    _end = time()
    print('Training Doc2Vec Model [{}]: {:,.02f} minutes'.format(iter_unit, (_end-_start)/60))
    print('# of Documents: {}'.format(len(d2v_model.model.docvecs)))
    return d2v_model


## Pairing
def paragraph_pairing(section2vec_model, paragraph2vec_model, topn):
    _start = time()

    section_list = [(tag, section2vec_model.model.docvecs[tag]) for tag in section2vec_model.model.docvecs.index2entity]
    pairs_section = utils.pairing(tag1_list=section_list, tag2_list=section_list)

    fdir_result = os.path.join(cfg['root'], cfg['fdir_result'])
    cnt = 0
    with tqdm(total=len(pairs_section)*topn) as pbar:
        for target_section_tag in pairs_section.keys():
            for paired_section_tag, score in pairs_section[target_section_tag][:topn]:
                target_paragraph_list = [(p.tag, paragraph2vec_model.model.infer_vector(doc_words=p.ngram, epochs=100)) for p in read.docs_included(iter_unit='paragraph', hyper_tag=target_section_tag)]
                paired_paragraph_list = [(p.tag, paragraph2vec_model.model.infer_vector(doc_words=p.ngram, epochs=100)) for p in read.docs_included(iter_unit='paragraph', hyper_tag=paired_section_tag)]
                pairs_paragraph = utils.pairing(tag1_list=target_paragraph_list, tag2_list=paired_paragraph_list)

                
                fname_result = '{}_{}.pk'.format(target_section_tag, paired_section_tag)
                fpath_result = os.path.join(fdir_result, 'paragraph_pairing/', fname_result)
                write.object(obj=pairs_paragraph, fpath=fpath_result)
                cnt += len(pairs_paragraph)
                pbar.update(1)

    _end = time()
    print('Paragraph Pairing: {:,} pairs from {:,} sections ({:,.02f} minutes)'.format(cnt, len(section_list), (_end-_start)/60))


## 
def export_paired_result(provision_info, min_pairing_score):
    fdir = os.path.join(cfg['root'], cfg['fdir_result'], 'paragraph_pairing/')
    flist = [fname for fname in os.listdir(fdir) if fname.startswith(provision_info)]

    fdir_result = os.path.join(cfg['root'], cfg['fdir_result'], 'paragraph_pairing_case_study/')

    for fname in flist:
        fpath = os.path.join(fdir, fname)
        with open(fpath, 'rb') as f:
            pairs = pk.load(f)

        data = defaultdict(list)
        for p_tag_target in sorted(pairs.keys(), reverse=False):
            p_tag_relevant, score = list(sorted(pairs[p_tag_target], key=lambda x:x[1], reverse=True))[0]
            
            data['target'].append(p_tag_target)
            if score > min_pairing_score:
                data['relevant'].append(p_tag_relevant)
                data['score'].append('{:.03f}'.format(score))
            else:
                data['relevant'].append('NO RELEVANT PARAGRAPH')
                data['score'].append('')

        fname_result = fname.replace('.pk', '.xlsx')
        fpath_result = os.path.join(fdir_result, fname_result)
        write.makedir(fpath_result)

        df = pd.DataFrame(data)
        df.to_excel(fpath_result, index=True, index_label='index')



if __name__ == '__main__':
    parameters_section_manual = {'vector_size': 500,
                                 'window': 10,
                                 'min_count': 10,
                                 'workers': 10,
                                 'dm': 1,
                                 'negative': 5,
                                 'epochs': 200,
                                 'dbow_words': 1}
    section2vec_model = doc2vec_embedding(iter_unit='section_manual',
                                          parameters=parameters_section_manual,
                                          train=False)
    
    parameters_paragraph = {'vector_size': 200,
                            'window': 10,
                            'min_count': 30,
                            'workers': 10,
                            'dm': 1,
                            'negative': 5,
                            'epochs': 200,
                            'dbow_words': 1}
    paragraph2vec_model = doc2vec_embedding(iter_unit='paragraph',
                                            parameters=parameters_paragraph,
                                            train=False)

    paragraph_pairing(section2vec_model=section2vec_model,
                      paragraph2vec_model=paragraph2vec_model,
                      topn=10)

    provision_info = 'Qatar_Qatar_2014_06_05'
    export_paired_result(provision_info=provision_info, min_pairing_score=0.8)