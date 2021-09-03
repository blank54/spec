#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os


class SpecPath:
    root = '/data/blank54/workspace/project/spec/'

    ## Data
    fdir_data = os.path.join(root, 'data')
    fdir_data_paragraph = os.path.join(fdir_data, 'paragraph')
    data_for_ner_bert_labeling = os.path.join(root, 'data/sentence/data_for_ner_bert_labeling.json')
    # fdir_data_section_manual = os.path.join(root, 'data/section/manual/')

    # ## Provision Pairing
    # fdir_ppr_data_exist = os.path.join(root, 'result/paragraph_pairing_casestudy_eval/')
    
    # ## Sentence Search
    # fdir_search_data = os.path.join(root, 'data/sentence_pair/')

    ## Corpus
    fdir_corpus = os.path.join(root, 'corpus')
    fdir_corpus_paragraph = os.path.join(fdir_corpus, 'paragraph')
    corpus_sentence = os.path.join(root, 'corpus/sentences.pk')